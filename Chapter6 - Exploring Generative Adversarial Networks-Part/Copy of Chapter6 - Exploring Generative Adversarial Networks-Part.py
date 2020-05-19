import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchsummary import summary
import numpy as np

#transforms.ToTensor() -> 데이터를 파이토치의 Tensor 형식으로바꾼다.
# transforms.Normalize((0.5, ), (0.5, )) --> 픽셀값 0 ~ 1 -> -1 ~ 1
transform = transforms.Compose([    #여러 trnasform들을 compose로 구성
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
])


batch_size = 128        # 배치사이즈  128사진마다
z_dim = 100

train_loader = torch.utils.data.DataLoader(
    dataset.MNIST('mnist', train = True, download = True, transform = transform),
    batch_size = batch_size,
    shuffle = True
)   # 데이터셋은 mnist 다운로드하고, 데이터를 한번에 batch_size만큼만 가져오는 dataloader를 만든다.

print("train_loader  :", train_loader)


# 데이터셋은 mnist

#가중치, bias 초기화하기
#초깃값을 모두 0으로 설정하면 오차역전파법에서 모든 가중치의 값이 똑같이 갱신되기 때문에 학습이 올바르게 이뤄지지 않습니다.

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# 만약에 cuda(Gpu)가 가능하지 않다면 cpu로 전환
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Generator_model 모델 생성
    # - ConvTranspose2d (이미지의 크기가 어떤 크기의 출력 이미지로 변경되는지에 대한 수식)
    # - BatchNorm2d  배치 정규화
    # - LeakyReLU   출력값이 0보다 높으면 그대로 놔두고, 0보다 낮으면 0.01으로 만든다

class Generator_model(nn.Module): # 생성자는 랜덤 벡터 z를 입력으로 받아 가짜 이미지를 출력한다.
    def __init__(self, z_dim):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 7 * 7)
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )
        #(batch_size x z_dim = 100) 크기의 랜덤 벡터를 받아
        # 이미지를 (batch_size x 256 x 7 x 7) 크기로 출력한다.
    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 256, 7, 7)
        return self.gen(x)
        # gen 새로운 이미지 생성
generator = Generator_model(z_dim).to(device)   #generator 모델 생성
generator.apply(weights_init)                      #genterator 초기화
summary(generator, (100, )) #genterator에 관한 요약된 통계데이터를 확인할 수 있습니다.


#판별해주는  모델 생성
    #- Conv2d 필터로 특징을 뽑아주는 컨볼루션(Convolution) 레이어
    #- sigmoid 0인지 1인지구분함으로써 분류하는 방식

# 구분자는 이미지를 입력으로 받아 이미지가 진짜인지 가짜인지 출력한다.
class Discriminator_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01)
        )
        self.fc = nn.Linear(2048, 1)

    #  이미지를 받아
    # 이미지가 진짜일 확률을 0~1 사이로 출력한다.
    def forward(self, input):
        x = self.disc(input)
        return F.sigmoid(self.fc(x.view(-1, 2048)))
        #sigmoid을 사용하여 0인지 1인지 판독을 해준다.
discriminator = Discriminator_model().to(device)    #객체 생성

discriminator.apply(weights_init)       #판별기 가중치초기화
summary(discriminator, (1, 28, 28))     #판별기에 관한 요약된 통계데이터를 확인할 수 있습니다.

criterion = nn.BCELoss()# 이진 분류에서 사용하는 크로스엔트로피 함수, 시그모이드 함수가 적용된 레이어를 사용해야한다,

# create a batch (whose size 64) of fixed noise vectors (z_dim=100)
fixed_noise = torch.randn(64, z_dim, device=device)

#Adam은 옵티마이저 -  Momentum의 장점 (가속도) + RMSProp의 장점 (각 매개변수마다 학습률 조절)

# 구분자의 매개 변수를 최적화하는 Adam optimizer
doptimizer = optim.Adam(discriminator.parameters())
# 생성자의 매개 변수를 최적화하는 Adam optimizer
goptimizer = optim.Adam(generator.parameters())

#라벨링 - 1 진짜 2 가짜 붙여줌
real_label, fake_label = 1, 0

image_list = []
g_losses = []
d_losses = []
iterations = 0
num_epochs = 50 #학습 50번

# for 구문을 돌려서 학습을 50번 시켜줌
for epoch in range(num_epochs):
    print(f'Epoch : | {epoch + 1:03} / {num_epochs:03} |')
    # 한번에 batch_size만큼 데이터를 가져온다.
    for i, data in enumerate(train_loader):

        discriminator.zero_grad()   #.backward()를 사용하면 미분값이 계속 누적된다. 미분값 변화도를 0으로 만들고,

        real_images = data[0].to(device)

        size = real_images.size(0)  #배치사이즈
        print("for each batch, size =" , size)
        # 이미지가 진짜일 때 정답 값은 1이고 가짜일 때는 0인 라벨링을 붙여준
        label = torch.full((size,), real_label, device=device)
        print("for each batch, label =" , label)

        d_output = discriminator(real_images).view(-1)  #진짜 이미지를 구분자에 넣는다.
        print("for each batch, d_output (real) =" , d_output)
        # 구분자의 출력값이 정답지인 1에서 멀수록 loss가 높아진다.
        derror_real = criterion(d_output, label)
        print("for each batch, derror_real =" , derror_real)
        derror_real.backward()   #역전파 단계 - 손실의 변화도를 계산

        noise = torch.randn(size, z_dim, device=device)
        fake_images = generator(noise)       # 생성자로 가짜 이미지를 생성한다.
        label.fill_(0)  # _: in-place-operation

        d_output = discriminator(fake_images.detach()).view(-1)#가짜 이미지를 구분자에 넣는다.
        print("for each batch, d_output(fake) =" , d_output)

        # 구분자의 출력값이 정답지인 1에서 멀수록 loss가 높아진다.
        derror_fake = criterion(d_output, label)
        print("for each batch, derror_fake =" , derror_fake)

        derror_fake.backward()  #역전파 단계 - 손실의 변화도를 계산

        # derror_total은 두 문제에서 계산된 loss의 합이다.
        derror_total = derror_real + derror_fake
        doptimizer.step()   #매개변수가 갱신

        generator.zero_grad()   #.backward()를 사용하면 미분값이 계속 누적된다. 미분값 변화도를 0으로 만들고,
        label.fill_(1)  # _: in-place-operation; the same as label.fill_(1)
        d_output = discriminator(fake_images).view(-1)
        gerror = criterion(d_output, label)
        gerror.backward()   #역전파 단계 - 손실의 변화도를 계산

        goptimizer.step()   #매개변수가 갱신

        if i % 50 == 0:
            print(
                f'| {i:03} / {len(train_loader):03} | G Loss: {gerror.item():.3f} | D Loss: {derror_total.item():.3f} |')
            g_losses.append(gerror.item())
            d_losses.append(derror_total.item())

        if (iterations % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(train_loader) - 1)):
            with torch.no_grad():
                fake_images = generator(fixed_noise).detach().cpu()
            image_list.append(vutils.make_grid(fake_images, padding=2, normalize=True))

        iterations += 1

plt.figure(figsize=(10,5))          #표  사이즈
plt.title("Generator and Discriminator Loss During Training")#표 제목
plt.plot(g_losses,label="Generator")
plt.plot(d_losses,label="Discriminator")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()      #표 이미지 그려주기


for image in image_list:
    plt.imshow(np.transpose(image,(1,2,0)))
    plt.show()