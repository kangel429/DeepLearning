from fastai.text import *

bs=64         #bs사이즈 64

path = untar_data(URLs.HUMAN_NUMBERS)   # HUMAN_NUMBERS 다운받고 압축풀기
path.ls()                               #경로 확인

def readnums(d): return [', '.join(o.strip() for o in open(path/d).readlines())]  #파일명을 집어넣어주면 - 열리면 모든 라인을 하나씩 하나씩 읽어서 o에서 집어넣어주고 그런 다음  양쪽 공백 지우기 - 리스트로 뽑은다음 가져올 때 ,를 뿥이고 가져와준다

train_txt = readnums('train.txt'); train_txt[0][:80]      #readnums함수로 읽은다음 저장해주고 0 ~80까지 찍어보기

valid_txt = readnums('valid.txt'); valid_txt[0][-80:]    #readnums함수로 읽은다음 저장해주고 -80에서 0까지 찍어보기

train = TextList(train_txt, path=path)    #TextList를 통해서 train_txt 읽어옵니다,
valid = TextList(valid_txt, path=path)    #TextList를 통해서 valid_txt 읽어옵니다,

src = ItemLists(path=path, train=train, valid=valid).label_for_lm() #ItemLists를 사용해서  train, valid 설정해주고 레이블은 내가 달지 않고 fastai 알아서 해줘
data = src.databunch(bs=bs)

train[0].text[:80]          #train 80개 까지 읽어보기

len(data.valid_ds[0][0].data)   #valid_ds의 data 몇개인지

data.bptt, len(data.valid_dl) #bptt(BackPropagation) 몇번째 단어까지 확인하는거냐? 70단어까지 확인해보겠다,  데이터 한번에 밀어넣는 단위가 몇개나 되는건지 3개

13017/70/bs       #3번만 넣어줌

it = iter(data.valid_dl)  #iter를 사용해서 data.valid_dl범위내에서
x1,y1 = next(it)    #다음번째
x2,y2 = next(it)    #그 다음번째
x3,y3 = next(it)    #그 그 다음번째 해서 읽어보기
it.close()

x1.numel()+x2.numel()+x3.numel()    #개수 다 더하니

x1.shape,y1.shape       #fastai   에서 랜덤값으로 나오기때문에 할 때마다 다른 값이 찍힘 bs=64  bptt70개

x2.shape,y2.shape   #fastai   에서 랜덤값으로 나오기때문에 할 때마다 다른 값이 찍힘 bs=64  bptt70개

x1[0,:]           #
y1[0,:]         #x1 값에서 하나씩 값이 밀려서 들어온다

v = data.valid_ds.vocab   #vocab리스트가 들어가 있음

v.textify(x1[0])      #x1 숫자데이터들을 텍스트화 시킴

v.textify(y1[0])      #마찬가지로 x1 값에서 하나씩 값이 밀려서 텍스트가 들어온다

v.textify(x2[0])
v.textify(x3[0])
v.textify(x1[1])
v.textify(x2[1])
v.textify(x3[1])
v.textify(x3[-1])

data.show_batch(ds_type=DatasetType.Valid)    #각 배치별로 데이터 읽어보기
data = src.databunch(bs=bs, bptt=3)     #databunch를 만들기  bptt=3 단어 3개만 보여주기,

x,y = data.one_batch()                # data.one_batch()해서
x.shape,y.shape             #값을 찍어두면 bs64  bptt=3 들어간 것을 확인할 수 있다

nv = len(v.itos); nv        #단어개수  itos 인티저를 스트링으로 바꿔줌

nh=64                   #히든뉴런개수 64

def loss4(input,target): return F.cross_entropy(input, target[:,-1])    #마지막 단어를 읽어서 cross_entropy   -----3개 단어를 집어넣으면 4번째 단어를 맞추기 위해
def acc4 (input,target): return accuracy(input, target[:,-1])   #마지막 단어를 읽어서 accuracy


class Model0(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv, nh)  # green arrow nv는 단어 종류, 64개 묶음  -- 백터 숫자들이 나옴
        self.h_h = nn.Linear(nh, nh)  # brown arrow nv는 단어 종류, 64개 묶음  히든레이어에서 히든레이어로
        self.h_o = nn.Linear(nh, nv)  # blue arrow nv는 단어 종류,  64개 묶음
        self.bn = nn.BatchNorm1d(nh)

    def forward(self, x):
        h = self.bn(F.relu(self.h_h(self.i_h(x[:, 0]))))  # i_h 인풋을 히든arrow로 바꾸는
        if x.shape[1] > 1:
            h = h + self.i_h(x[:, 1])  # Embedding사용해서 인풋에서 히든레이어로 변환에서 h를 더해줌
            h = self.bn(F.relu(self.h_h(h)))
        if x.shape[1] > 2:
            h = h + self.i_h(x[:, 2])  # 히든레이어에서 히든레이어로 변환
            h = self.bn(F.relu(self.h_h(h)))
        return self.h_o(h)  # 마지막 단어를 뽑아낼 수 있음

learn = Learner(data, Model0(), loss_func=loss4, metrics=acc4)


learn.fit_one_cycle(6, 1e-4)     #6번 학습


class Model1(nn.Module):  # n번째 단어를 추정
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv, nh)  # green arrow nv는 단어 종류, 64개 묶음  -- 백터 숫자들이 나옴
        self.h_h = nn.Linear(nh, nh)  # brown arrow  nv는 단어 종류, 64개 묶음  히든레이어에서 히든레이어로
        self.h_o = nn.Linear(nh, nv)  # blue arrow  nv는 단어 종류,  64개 묶음
        self.bn = nn.BatchNorm1d(nh)  # BatchNorm1d

    def forward(self, x):
        h = torch.zeros(x.shape[0], nh).to(device=x.device)
        for i in range(x.shape[1]):  # 최대 몇개가지 받아서 할 수 있는지 적어서
            h = h + self.i_h(x[:, i])  # Embedding사용해서 인풋에서 히든레이어로 변환에서 h를 더해줌
            h = self.bn(F.relu(self.h_h(h)))  # 히든레이어에서 히든레이어로 변환
        return self.h_o(h)  # 마지막 단어를 뽑아낼 수 있음

learn = Learner(data, Model1(), loss_func=loss4, metrics=acc4)

learn.fit_one_cycle(6, 1e-4)      #6번 학습

data = src.databunch(bs=bs, bptt=20)        #databunch를 만들기  bptt=20 단어 20개만 보여주기,

x,y = data.one_batch()
x.shape,y.shape       #bs 64  bptt=20


class Model2(nn.Module):  # 여러단어를 추정할 수 있게  2번째 단어부터 n번째 단어 추정
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv, nh)
        self.h_h = nn.Linear(nh, nh)
        self.h_o = nn.Linear(nh, nv)
        self.bn = nn.BatchNorm1d(nh)

    def forward(self, x):
        h = torch.zeros(x.shape[0], nh).to(device=x.device)
        res = []
        for i in range(x.shape[1]):
            h = h + self.i_h(x[:, i])
            h = F.relu(self.h_h(h))
            res.append(self.h_o(self.bn(h)))
        return torch.stack(res, dim=1)  # dim=1 축을 중심으로 stack을 이용해서 하나의 텐서로 만들어줌

    learn = Learner(data, Model2(), metrics=accuracy)  # 모든 단어를 볼 수 있게 lossfunc 지정  ,accuracy

    learn.fit_one_cycle(10, 1e-4, pct_start=0.1)  # 10번 학습


class Model3(nn.Module):  # 여러단어를 추정할 수 있게  2번째 단어부터 n번째 단어 추정
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv, nh)
        self.h_h = nn.Linear(nh, nh)
        self.h_o = nn.Linear(nh, nv)
        self.bn = nn.BatchNorm1d(nh)
        self.h = torch.zeros(bs, nh).cuda()

    def forward(self, x):
        res = []
        h = self.h
        for i in range(x.shape[1]):
            h = h + self.i_h(x[:, i])
            h = F.relu(self.h_h(h))
            res.append(self.bn(h))
        self.h = h.detach()  # self.h 저장 나중에는 h 재활용
        res = torch.stack(res, dim=1)
        res = self.h_o(res)
        return res


learn = Learner(data, Model3(), metrics=accuracy)

learn.fit_one_cycle(20, 3e-3)


class Model4(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv, nh)
        self.rnn = nn.RNN(nh, nh, batch_first=True)  # RNN사용해서  batch_first 배치 먼저 지정
        self.h_o = nn.Linear(nh, nv)
        self.bn = BatchNorm1dFlat(nh)
        self.h = torch.zeros(1, bs, nh).cuda()

    def forward(self, x):
        res, h = self.rnn(self.i_h(x), self.h)  # 인풋에서 히든, 히든 읽어오고
        self.h = h.detach()
        return self.h_o(self.bn(res))

learn = Learner(data, Model4(), metrics=accuracy)
learn.fit_one_cycle(20, 3e-3)


class Model5(nn.Module):  # GRU 사용 얼마나 옛날 것을 잊어버리고 기억할건지 패턴을 찾는다
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv, nh)
        self.rnn = nn.GRU(nh, nh, 2, batch_first=True)
        self.h_o = nn.Linear(nh, nv)
        self.bn = BatchNorm1dFlat(nh)
        self.h = torch.zeros(2, bs, nh).cuda()

    def forward(self, x):
        res, h = self.rnn(self.i_h(x), self.h)
        self.h = h.detach()
        return self.h_o(self.bn(res))

learn = Learner(data, Model5(), metrics=accuracy)

learn.fit_one_cycle(10, 1e-2)