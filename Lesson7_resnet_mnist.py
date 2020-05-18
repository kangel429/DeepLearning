from fastai.vision import *

path = untar_data(URLs.MNIST)

print(path.ls())

il = ImageList.from_folder(path, convert_mode='L') #이미지 리스트   이 이미지는  convert_mode='L' 흑백모드
il.items[0]       #이 이미지의 아이템은 -- 33125.png

defaults.cmap='binary'

il[0].show()

sd = il.split_by_folder(train='training', valid='testing')#훈련 데이터와 검증 데이터 폴더를 나누고
print(sd)

print((path/'training').ls())
ll = sd.label_from_folder()
print(ll)

x,y = ll.train[0]             #첫번째 있는 데이터 가지고 오기

x.show()
print(y,x.shape)              #첫번째 있는 데이터 찍어보기

tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], []) #무작위의 패딩
ll = ll.transform(tfms)       #변형
bs = 128

# not using imagenet_stats because not using pretrained model
data = ll.databunch(bs=bs).normalize()

x,y = data.train_ds[0]    #노멀라이징 해서 다시 첫번째 있는 데이터 가지고 오기

x.show()
print(y)

def _plot(i,j,ax): data.train_ds[0][0].show(ax, cmap='gray')  #무작위 패딩을 했기 때문에 각각의 0이 다른 위치에 있다
plot_multi(_plot, 3, 3, figsize=(8,8))              #그림을 멀티로 보여줌 3*3


xb,yb = data.one_batch()
xb.shape,yb.shape
data.show_batch(rows=3, figsize=(5,5))

def conv(ni,nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1) # 커널 크기 3 보폭 2 패딩 1  컨볼 루션이있을 때마다 한 픽셀 매번 두 단계 씩 점프합니다. 그리드 크기를 반으로 줄입니다.

model = nn.Sequential(
    conv(1, 8), # 14    conv함수를 이용해서 반복 작업해줌
    nn.BatchNorm2d(8),
    nn.ReLU(),
    conv(8, 16), # 7
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 32), # 4
    nn.BatchNorm2d(32),
    nn.ReLU(),
    conv(32, 16), # 2
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 10), # 1
    nn.BatchNorm2d(10),
    Flatten()     # remove (1,1) grid
)

learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)     #모델 만듬

print(learn.summary())

xb = xb.cuda()    #GPU에 팝업 xb.cuda()하고 배치를 모델에 전달

model(xb).shape

learn.lr_find(end_lr=100)

learn.recorder.plot()

learn.fit_one_cycle(3, max_lr=0.1)

def conv2(ni,nf): return conv_layer(ni,nf,stride=2) #전환, 배치 규범, ReLU의 조합
model = nn.Sequential(
    conv2(1, 8),   # 14
    conv2(8, 16),  # 7
    conv2(16, 32), # 4
    conv2(32, 16), # 2
    conv2(16, 10), # 1
    Flatten()      # remove (1,1) grid
)
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)

learn.fit_one_cycle(10, max_lr=0.1)


class ResBlock(nn.Module):  # nn 모듈의 인자값을 집어넣을 수 있는 객체 만들기  ---- 이 함수를 통해서 좀 더 정확한 결과물을 얻을 수 있다
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer(nf, nf)  # 레이어 2개 만듬
        self.conv2 = conv_layer(nf, nf)

    def forward(self, x): return x + self.conv2(self.conv1(x))

help(res_block)
model = nn.Sequential(
    conv2(1, 8),
    res_block(8),
    conv2(8, 16),
    res_block(16),
    conv2(16, 32),
    res_block(32),
    conv2(32, 16),
    res_block(16),
    conv2(16, 10),
    Flatten()
)

def conv_and_res(ni,nf): return nn.Sequential(conv2(ni, nf), res_block(nf))

model = nn.Sequential(
    conv_and_res(1, 8),
    conv_and_res(8, 16),
    conv_and_res(16, 32),
    conv_and_res(32, 16),
    conv2(16, 10),
    Flatten()
)


learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)

learn.lr_find(end_lr=100)
learn.recorder.plot()

learn.fit_one_cycle(12, max_lr=0.05)

print(learn.summary())



