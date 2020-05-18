import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *
import os
import sys
currentPath = os.getcwd()
path = untar_data(URLs.PETS)    #애완동물 다운
path_hr = path/'images'
path_lr = path/'crappy'
print(currentPath)
crappify_dir = currentPath+'/zh-nbs/'
sys.path.insert(0, crappify_dir)

import crappify

crappify

il = ImageList.from_folder(path_hr)     #임포트 c
parallel(crappify.crappifier(path_lr, path_hr), il.items) #병렬로 사용하기 때문에 시간 단축됨

bs,size=32, 128
# bs,size = 24,160
#bs,size = 8,256
arch = models.resnet34

arch = models.resnet34
src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42) #이미지 리스트


def get_data(bs,size):                        #32장마다 이미지 얻기
    data = (src.label_from_func(lambda x: path_hr/x.name)
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data

data_gen = get_data(bs,size)

data_gen.show_batch(4)

wd = 1e-3

y_range = (-3.,3.)

loss_gen = MSELossFlat()

def create_gen_learner():     #모델계층만들기
    return unet_learner(data_gen, arch, wd=wd, blur=True, norm_type=NormType.Weight,
                         self_attention=True, y_range=y_range, loss_func=loss_gen)


learn_gen = create_gen_learner()

learn_gen.fit_one_cycle(2, pct_start=0.8)   #학습  0.8부터 속도로 학습

learn_gen.unfreeze()

learn_gen.fit_one_cycle(3, slice(1e-6,1e-3))    #다시 학습

learn_gen.show_results(rows=4)      # 입력 / 예측 /  목표

learn_gen.save('gen-pre2')  # 결과값을 보면 아직 목표에 미치지 못했다 해결방법  ---> gan 이다

learn_gen.load('gen-pre2');

name_gen = 'image_gen'
path_gen = path/name_gen

path_gen.mkdir(exist_ok=True)


def save_preds(dl):  # 생성된 이미지 저장
    i = 0
    names = dl.dataset.items

    for b in dl:
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen / names[i].name)
            i += 1

save_preds(data_gen.fix_dl)
PIL.Image.open(path_gen.ls()[0])

learn_gen=None
gc.collect()

def get_crit_data(classes, bs, size):
    src = ImageList.from_folder(path, include=classes).split_by_rand_pct(0.1, seed=42)
    ll = src.label_from_folder(classes=classes)
    data = (ll.transform(get_transforms(max_zoom=2.), size=size)
           .databunch(bs=bs).normalize(imagenet_stats))
    data.c = 3
    return data

data_crit = get_crit_data([name_gen, 'images'], bs=bs, size=size)

data_crit.show_batch(rows=3, ds_type=DatasetType.Train, imgsize=3)

loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())

def create_critic_learner(data, metrics):
    return Learner(data, gan_critic(), metrics=metrics, loss_func=loss_critic, wd=wd)

learn_critic = create_critic_learner(data_crit, accuracy_thresh_expand)

learn_critic.fit_one_cycle(6, 1e-3)

learn_critic.save('critic-pre2')

learn_crit=None
learn_gen=None
gc.collect()

data_crit = get_crit_data(['crappy', 'images'], bs=bs, size=size) #평가를 초기화

learn_crit = create_critic_learner(data_crit, metrics=None).load('critic-pre2')

learn_gen = create_gen_learner().load('gen-pre2')     #gan 모델 사용

switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65) #가짜인지 진짜인지 판별하여 돌려보내는 역할
learn = GANLearner.from_learners(learn_gen, learn_crit, weights_gen=(1.,50.), show_img=False, switcher=switcher,
                                 opt_func=partial(optim.Adam, betas=(0.,0.99)), wd=wd) #gan모델 계층을 사용
learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))

lr = 1e-4

learn.fit(40,lr)

learn.save('gan-1c')

learn.data=get_data(16,192)
learn.fit(10,lr/2)
learn.show_results(rows=16)