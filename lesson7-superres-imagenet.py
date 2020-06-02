import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *

from torchvision.models import vgg16_bn #뉴럴네트워크 vgg-16

path = Path('data/imagenet')    #경로 설정
path_hr = path/'train'          #고해상도 해상도 이미지저장될 경로
path_lr = path/'small-64/train' #64*64 해상도 이미지저장될 경로
path_mr = path/'small-256/train'#256*256 해상도 이미지저장될 경로

path_pets = untar_data(URLs.PETS)   #펫 데이터 셋을 사용

il = ImageList.from_folder(path_hr)     #이 폴더에서 고해상도 이미지를 읽어옵니다

def resize_one(fn, i, path, size):       #사이즈 조정
    dest = path/fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True) # 이 경로 없으면 만들어줌
    img = PIL.Image.open(fn)                        #fn 이미지 가져오기
    targ_sz = resize_to(img, size, use_min=True)   #fastai 라이브러리에서 제공되는 resize로, 최소 size로 비율에 맞게 줄어듬
    img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
    img.save(dest, quality=60)                    #사이즈 조정된 이미지 저장

assert path.exists(), f"need imagenet dataset @ {path}"
#parallel 반복된 명령어 사용할 때 - 사이즈 조정
sets = [(path_lr, 64), (path_mr, 256)]
for p,size in sets:
    if not p.exists():
        print(f"resizing to {size} into {p}")
        parallel(partial(resize_one, path=p, size=size), il.items)

free = gpu_mem_get_free_no_cache()
# the max size of the test image depends on the available GPU RAM
if free > 8200: bs,size=16,256
else:           bs,size=8,256
print(f"using bs={bs}, size={size}, have {free}MB of GPU RAM free")

arch = models.resnet34      #models.resnet34 기본 모델 사용
# sample = 0.1
sample = False

tfms = get_transforms()

src = ImageImageList.from_folder(path_lr)   #저해상도 이미지를 로딩해주고

if sample: src = src.filter_by_rand(sample, seed=42)
src = src.split_by_rand_pct(0.1, seed=42)   #10퍼센트 검증이미지로 사용하겠다

def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr/x.relative_to(path_lr))
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data
#레이블은 저해상도 이미지를 받아서 path_hr파일 이미지로 리턴해주도록 그래서 레이블 이름 사용
#줌을 2배로
#데이터셋을 생성해서 imagenet_stats사용해서 노멀라이징 시켜줍니다  x는 저화질 이미지 y는 고화질 이미지  tfm_y=True, do_y= turue 사용

data = get_data(bs,size)

def gram_matrix(x):
    n,c,h,w = x.size()      #n 데이터 개수 (배치), c채널, h세로, w가로 얻어옵니다
    x = x.view(n, c, -1)    # x.view --> reshape해줌 -1을 하면 h하고 w 곱한 값이 자동으로 들어감
    return (x @ x.transpose(1,2))/(c*h*w)   #1번 축하고 2번축을 transpose 하고 메트릭스 해줍니다 ---> 3채널 각각 값 9개의 값이 나옴 -->이런 값이 비슷하게 나오면 스타일을 비슷하게 생성할 수 있음


vgg_m = vgg16_bn(True).features.cuda().eval()    #vgg16 사용해서 features값 컨블루션만 사용하겠다는 말 그래서 쿠다 gpu사용하고 eval()사용하겠다
requires_grad(vgg_m, False)                 #학습은 안시킬거니까 grad 사용할 필요없다 false
blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]#사이즈가 줄어드는 순간 직전에 MaxPool2d 잡아냄

base_loss = F.l1_loss           #손실 함수 정의 뺀다음에 절대값 취한다음에 평균 구한 값


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel', ] + [f'feat_{i}' for i in range(len(layer_ids))
                                           ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):  # make_features 값을 뽑아냄
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):  # input 저화질  target 목표값 ----- loss 계산해서  값을 리턴시킴
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)  # make_features 값을 뽑아냄
        self.feat_losses = [base_loss(input, target)]  # F.l1_loss로 계산해줌
        self.feat_losses += [base_loss(f_in, f_out) * w  # 레이어별로 가중치를 다르게 값을 주기해서 w을 곱하기 함
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]  # f_in, f_out 둘다 계산해줌
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out)) * w ** 2 * 5e3
                             # gram_matrix계산해주고 레이어별로 가중치를 다르게 값을 주기해서 w을 곱하기 함
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))  # dict형태로 저장
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()

feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])

wd = 1e-3
learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,    #resnet34 사용  feat_loss 사용 LossMetrics --> 메트릭스를 주기적으로 프린트를 할 수 있다
                     blur=True, norm_type=NormType.Weight)
gc.collect();
learn.unfreeze()

learn.load((path_pets/'small-96'/'models'/'2b').absolute()); #학습한 모델 불러옴

learn.fit_one_cycle(1, slice(1e-6,1e-4)) # 학습시킴

learn.save('imagenet') #저장

learn.show_results(rows=3, imgsize=5) #학습한 거 보여줌

learn.recorder.plot_losses()    #로스그래프 보여

_=learn.load('imagenet') #모델 학습하거 다시 불러옴

data_mr = (ImageImageList.from_folder(path_mr).split_by_rand_pct(0.1, seed=42)
          .label_from_func(lambda x: path_hr/x.relative_to(path_mr))
          .transform(get_transforms(), size=(820,1024), tfm_y=True)
          .databunch(bs=2).normalize(imagenet_stats, do_y=True))

#size=(820,1024) 이 목표로 사이즈 해상도를 좋게 만들 예정
learn.data = data_mr
fn = path_pets/'other'/'dropout.jpg'    #테스트할 사진 가져옴
img = open_image(fn); img.shape #테스트 이미지 크
_,img_hr,b = learn.predict(img)

show_image(img, figsize=(18,15), interpolation='nearest'); #저화질 이미지

Image(img_hr).show(figsize=(18,15)) #고화질 이미지