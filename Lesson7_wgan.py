from fastai.vision import *
from fastai.vision.gan import *

path = untar_data(URLs.LSUN_BEDROOMS)   #LSUN_BEDROOMS 데이터 가져와서 압축을 풀어줌

def get_data(bs, size):       #GANItemList를 폴더를 통해서 가지고 옴	노이즈 100개를 사용, 검증 훈련셋 나누지 않겠다, 라벨 필요없다. tfm_y에 해당되는 트렌스폼은 적당히 크롭해줌 (크기가 크면 크롭 크기가 작으면 패딩), 표준편차를 rgb 0.5라고 가정하고 노멀라이징 시켜줌
    return (GANItemList.from_folder(path, noise_sz=100)
               .split_none()
               .label_from_func(noop)
               .transform(tfms=[[crop_pad(size=size, row_pct=(0,1), col_pct=(0,1))], []], size=size, tfm_y=True)
               .databunch(bs=bs)
               .normalize(stats = [torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])], do_x=False, do_y=True))


data = get_data(128, 64)    #bs 128, size 64*64
data.show_batch(rows=5)   #데이터 보여주기

generator = basic_generator(in_size=64, n_channels=3, n_extra_layers=1)   #basic generator 만들어줌 (size = 64, rgb 3 채널), 중간에 컨버셜 레이러 몇개를 넣어줄 거냐 1개
critic    = basic_critic   (in_size=64, n_channels=3, n_extra_layers=1)   #basic_critic 만들어줌 (size = 64, rgb 3 채널), 중간에 컨버셜 레이러 몇개를 넣어줄 거냐 1개

learn = GANLearner.wgan(data, generator, critic, switch_eval=False,
                        opt_func = partial(optim.Adam, betas = (0.,0.99)), wd=0.) #wgan (gan아케텍쳐그중에 하나 )구현된 모델링을 가져옴. generator, critic 정의한거 집어넣고,  옵티마이저 아담 사용 switch_eval=false는 generator와 critic이 왔다갔다 할 때 트레인모드 이벨류레이션 모드 스위치(왔다갔다)하지 말라는 의미

learn.fit(30,2e-4)      #30번 훈련

learn.gan_trainer.switch(gen_mode=True)                   #gen_mode=True 바꿔주고
learn.show_results(ds_type=DatasetType.Train, rows=16, figsize=(8,8)) #다시한번 학습결과를 찍어봄