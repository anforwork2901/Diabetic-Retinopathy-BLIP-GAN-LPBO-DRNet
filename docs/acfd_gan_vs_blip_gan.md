# So sanh cau truc ACFD-GAN va BLIP-GAN

Tai lieu nay tom tat su khac nhau giua mo hinh ACFD-GAN trong paper *High-quality synthetic image with ACFD-GAN for enhanced diabetic retinopathy grading* va phien ban BLIP-GAN dang duoc cai tien trong project tai `notebooks/experiments/BLIP_GAN/`.

## 1. Tong quan

ACFD-GAN la mo hinh GAN co dieu kien duoc thiet ke de tong hop anh vong mac benh vong mac dai thao duong theo tung muc do benh. Muc tieu chinh cua ACFD-GAN la giai quyet mat can bang lop trong cac bo du lieu DR bang cach sinh anh tong hop chat luong cao cho cac lop thieu mau.

BLIP-GAN trong project nay duoc phat trien dua tren nen tang ACFD-GAN, nhung bo sung them cac co che tien xu ly mask, bao toan nen vong mac va sinh anh tap trung vao vung ton thuong. Vi vay, BLIP-GAN co the duoc xem la phien ban tai cau truc/mo rong cua ACFD-GAN theo huong lesion-aware va background-preserving.

> Luu y: trong code hien tai, mot so class van giu ten `ACFDGenerator`, `ACFDDiscriminator`. Tuy nhien, pipeline BLIP-GAN da duoc bo sung them cac co che rieng nhu weak-mask soft fusion, lesion inpainting, background preservation va post-hoc multi-seed FID.

## 2. Cau truc ACFD-GAN trong paper

ACFD-GAN gom cac thanh phan chinh:

1. WAE - Wasserstein AutoEncoder
2. Generator dang U-Net
3. LRDB - Lightweight Residual Dense Block
4. ACFF - Adaptive Cross-layer Feature Fusion
5. AMM - Adaptive Modulation Module
6. PatchGAN Discriminator
7. WMA-FID model selection

### 2.1 WAE latent conditioning

Paper su dung WAE de hoc latent representation cua anh that. Latent vector tu WAE duoc ket hop voi random noise de dieu khien qua trinh sinh anh.

Muc dich:

- Giu lai thong tin cau truc tong quat cua anh vong mac.
- Tao su da dang thong qua random noise.
- Giam kha nang sinh anh vo nghia hoac mat cau truc toan cuc.

Trong code BLIP-GAN, phan nay duoc cai dat bang:

- `WAEEncoder`
- `WAEDecoder`
- `WAE`
- `mmd_loss`

WAE duoc train truoc, sau do encoder duoc dong bang de lay latent vector lam dieu kien cho Generator.

### 2.2 Generator dang U-Net

Generator cua ACFD-GAN co dang encoder-decoder tuong tu U-Net:

```text
Mask / condition input
    -> Encoder voi LRDB + downsample
    -> Bottleneck ket hop latent z
    -> Decoder voi upsample
    -> ACFF skip fusion
    -> AMM modulation
    -> Synthetic fundus image
```

Generator khong chi sinh anh tu noise ma con nhan thong tin dieu kien tu mask va latent representation. Dieu nay giup anh sinh ra co lien ket tot hon voi cau truc vong mac va vung ton thuong.

### 2.3 LRDB - Lightweight Residual Dense Block

LRDB la khoi trich xuat dac trung cuc bo nhe. Trong paper, LRDB duoc dung de thay the cac convolution block nang hon.

Chuc nang:

- Trich xuat dac trung cuc bo nhieu cap.
- Giu thong tin chi tiet nho nhu vi phinh mach, xuat huyet, tiet cung.
- Giam chi phi tinh toan so voi RDB day du.

Trong code BLIP-GAN, LRDB duoc cai dat theo huong:

- `1x1 Conv` de giam chieu kenh.
- Cac `DepthwiseSeparableConv` de hoc dac trung nhe.
- Ket noi dense dang cong tich luy.
- `1x1 Conv` de hop nhat dac trung va residual fusion.

### 2.4 ACFF - Adaptive Cross-layer Feature Fusion

ACFF la diem quan trong cua ACFD-GAN. Module nay hop nhat feature tu encoder va decoder bang attention.

Chuc nang:

- Ket hop shallow feature co nhieu chi tiet bien/cau truc.
- Ket hop deep feature co thong tin ngu nghia cao.
- Su dung channel attention va spatial attention de gan trong so cho feature quan trong.
- Giam mat chi tiet lesion nho khi decoder phuc hoi anh.

Trong code BLIP-GAN, `ACFF` gom:

- Channel attention tu avg/max pooled fused feature.
- Spatial attention tu avg/max theo chieu kenh.
- Conv hop nhat feature sau attention.

### 2.5 AMM - Adaptive Modulation Module

AMM dung latent vector va mask de dieu bien feature trong decoder.

Chuc nang:

- Dua thong tin mask vao qua trinh sinh anh.
- Ket hop WAE latent va mask latent.
- Sinh tham so scale/shift theo tung feature map, gan voi y tuong AdaIN.
- Tang tinh da dang nhung van giu cau truc tong quat.

Trong code, `AMM` gom:

- InstanceNorm.
- Mask encoder.
- Linear projection tu mask sang latent.
- Mapping network tao `gamma` va `beta`.
- Dieu bien feature decoder bang `gamma`, `beta`.

### 2.6 Discriminator

ACFD-GAN su dung PatchGAN discriminator. Discriminator khong chi nhin anh sinh ra ma con nhin ca dieu kien mask.

```text
Input to D = concat(image, mask)
```

Chuc nang:

- Danh gia tinh that/gia theo patch cuc bo.
- Phu hop voi anh y khoa vi lesion la cac chi tiet nho, phan bo cuc bo.
- Buoc Generator sinh chi tiet texture va lesion thuyet phuc hon.

Trong BLIP-GAN, discriminator duoc cai dat voi:

- 5 lop PatchGAN.
- 3 stride-2 layer va 2 stride-1 layer.
- Spectral normalization tren convolution.
- Dau vao la `image + mask`.

### 2.7 Loss function va model selection

ACFD-GAN dung cac loss chinh:

```text
Generator loss = adversarial loss + lambda_L1 * L1 + lambda_VGG * perceptual loss
```

Trong code:

- `ADV_LOSS_TYPE = "bce"`
- `LAMBDA_L1 = 10.0`
- `LAMBDA_VGG = 10.0`
- VGG19 perceptual loss de giu texture/structure.
- FID va WMA-FID de chon checkpoint sinh anh tot.

Paper dung FID va MSE de danh gia chat luong anh sinh. Ngoai ra, anh sinh duoc dua vao classifier de kiem tra tac dong len accuracy, QWK, AUC va confusion matrix.

## 3. BLIP-GAN trong project

BLIP-GAN trong project giu lai khung chinh cua ACFD-GAN, nhung bo sung cac co che thuc dung hon de phu hop voi anh fundus va bai toan DR:

```text
Real fundus image
    -> Weak lesion/vessel/OD mask generation
    -> WAE latent extraction
    -> Random noise
    -> ACFD-style Generator
    -> Lesion-region inpainting
    -> Background-preserved synthetic image
    -> FID/WMA-FID/post-hoc FID selection
```

### 3.1 Weak-mask soft fusion

Khac voi paper co the dua tren mask/co che trich lesion ly tuong hon, project su dung weak mask duoc tao tu anh that bang OpenCV/skimage.

Trong code:

- `make_weak_mask`
- `MASK_MODE = "softfusion_v2_rich"`
- Frangi vesselness.
- Hybrid Canny.
- Kenh sang/toi de bat hard exudates, hemorrhage/microaneurysm.
- Erode fundus mask de giam anh huong vien den.

Y nghia:

- Khong can annotation lesion thu cong.
- Van tao duoc dieu kien cho Generator biet nen sinh lesion o dau.
- Giu lai cac tin hieu yeu cua mach mau va vung ton thuong nho.

### 3.2 Lesion-region inpainting

Day la diem khac biet lon cua BLIP-GAN so voi ACFD-GAN nen tang.

Thay vi de Generator sinh lai toan bo anh, BLIP-GAN chi uu tien thay doi vung lesion-heavy, con nen vong mac on dinh duoc copy tu anh goc.

Trong code:

```python
return base_img * (1.0 - alpha) + raw_fake * alpha
```

Trong do `alpha` duoc tao tu mask lesion:

- Bright lesion channel.
- Dark lesion channel.
- Blur de bien chuyen mem.
- Gioi han alpha de tranh pha huy toan bo anh.

Y nghia:

- Giam artifact o nen vong mac.
- Giu optic disc, mach mau, mau nen va vung khong benh on dinh hon.
- Buoc Generator tap trung vao phan quan trong nhat: lesion.
- Giam nguy co sinh anh dep nhung sai giai phau.

### 3.3 Background preservation loss

BLIP-GAN bo sung them co che phat loi o vung nen:

```text
background preserve loss = L1(fake * background, real * background)
```

Trong code:

- `BACKGROUND_PRESERVE_WEIGHT = 2.0`
- `bg_w = 1.0 - lesion_alpha_from_mask(...)`
- Loss nen duoc cong vao Generator loss.

Y nghia:

- Neu vung nen khong co lesion, Generator khong nen thay doi manh.
- Giam cac bien dang mau sac va texture khong can thiet.
- Giu synthetic image gan voi anh that hon ve background.

### 3.4 Candidate filtering / multi-candidate generation

BLIP-GAN co co che sinh nhieu candidate cho mot mask trong mot so buoc FID/generation:

- `FID_CANDIDATES_PER_MASK`
- `FID_CANDIDATE_CHUNK`
- Chon candidate dua tren feature/perceptual distance.

Y nghia:

- Giam rui ro mot lan sample noise sinh anh xau.
- Tang on dinh cua FID.
- Phu hop khi so luong anh real moi class it.

### 3.5 Multi-seed post-hoc FID

BLIP-GAN khong chi tinh FID mot lan, ma co them post-hoc FID voi nhieu seed:

- `POSTHOC_FID_SEEDS = [42, 123, 2026]`
- Tinh mean/std FID.

Y nghia:

- Giam phu thuoc vao mot seed may man.
- Bao cao FID on dinh hon.
- De thuyet phuc hon khi so sanh chat luong anh sinh.

### 3.6 Pipeline cho APTOS va Messidor

BLIP-GAN co hai notebook rieng:

- `BLIP_GAN_APTOS.ipynb`
- `BLIP_GAN_Messidor.ipynb`

Ca hai giu chung architecture chinh, nhung duong dan, dataset loader va tag thuc nghiem duoc dieu chinh cho tung dataset.

## 4. Bang so sanh ACFD-GAN va BLIP-GAN

| Thanh phan | ACFD-GAN paper | BLIP-GAN trong project |
|---|---|---|
| Muc tieu | Sinh anh DR chat luong cao de can bang lop | Sinh anh DR chat luong cao, uu tien vung lesion va giu nen vong mac on dinh |
| Input dieu kien | Mask/condition + WAE latent + noise | Weak mask tu anh that + WAE latent + noise + base image cho inpainting |
| Mask | Huong den structural/lesion mask theo paper | Weak-mask `softfusion_v2_rich` tu Frangi, Canny, bright/dark lesion cues |
| Generator | U-Net voi LRDB, ACFF, AMM | Giu U-Net ACFD-style, them lesion-region inpainting khi synthesize |
| Encoder block | LRDB | LRDB voi depthwise separable conv de nhe hon |
| Skip fusion | ACFF channel + spatial attention | ACFF channel + spatial attention |
| Modulation | AMM voi WAE latent + random noise | AMM voi WAE latent + random noise + mask latent |
| Discriminator | Conditional PatchGAN | Conditional PatchGAN co spectral normalization, input `image + mask` |
| Loss | Adv + L1 + perceptual | Adv + L1 + VGG perceptual + background preserve loss |
| Vung sinh anh | Co xu huong sinh toan bo anh | Sinh/pha tron chu yeu o vung lesion-heavy, giu nen that |
| Chon checkpoint | FID/WMA-FID | FID, WMA-FID, multi-seed post-hoc FID |
| Kha nang ung dung | Tang du lieu DR noi chung | Tang du lieu DR voi uu tien lesion realism va background stability |

## 5. Diem manh cua BLIP-GAN

### 5.1 Tap trung vao vung benh thay vi sinh lai toan bo anh

Anh fundus co nhieu vung nen on dinh: mau nen, mach mau, optic disc, vien anh. Neu Generator sinh lai toan bo anh, no de tao artifact lam classifier hoc sai. BLIP-GAN giai quyet bang lesion-region inpainting: vung khong lien quan duoc giu tu anh that, vung lesion moi duoc dieu chinh/sinh them.

Day la diem rat phu hop voi DR vi thong tin quyet dinh grade thuong nam o cac ton thuong nho.

### 5.2 Giam artifact nen vong mac

Background preservation loss ep anh sinh khong thay doi manh o vung khong lesion. Dieu nay giup synthetic image:

- It bi bien mau nen.
- It sai cau truc mach mau.
- It tao artifact quanh optic disc.
- Giu tinh y khoa tot hon so voi GAN sinh toan anh.

### 5.3 Khong can annotation lesion thu cong

BLIP-GAN dung weak mask tu xu ly anh, thay vi can mask lesion duoc bac si gan nhan.

Day la loi the lon trong do an vi:

- De ap dung cho APTOS va Messidor.
- Giam chi phi annotation.
- Van cung cap dieu kien lesion-aware cho Generator.

### 5.4 Giu duoc cac diem manh cua ACFD-GAN

BLIP-GAN khong bo di ACFD-GAN ma giu lai cac thanh phan quan trong:

- WAE latent representation.
- LRDB.
- ACFF.
- AMM.
- PatchGAN.
- VGG perceptual loss.
- WMA-FID.

Do do, BLIP-GAN vua ke thua nen tang paper, vua bo sung cac co che thuc nghiem de on dinh hon tren du lieu DR thuc te.

### 5.5 Danh gia chat luong anh chat che hon

BLIP-GAN co them:

- FID theo epoch.
- WMA-FID.
- Post-hoc FID voi nhieu seed.
- MSE theo epoch.
- G/D losses.
- Preview anh real/mask/fake.

Nhung thanh phan nay giup kiem soat chat luong anh sinh, tranh chon checkpoint dua tren mot epoch may man.

### 5.6 Phu hop voi muc tieu tang du lieu hiem

Trong DR, cac class 3 va 4 thuong it mau. BLIP-GAN co the sinh anh theo tung class, dac biet la cac lop benh nang, de ho tro classifier hoc duoc lesion pattern tot hon.

Khi ket hop voi pipeline classification, synthetic data tu BLIP-GAN co vai tro:

- Tang so mau cho lop hiem.
- Giam mat can bang lop.
- Bo sung bien the lesion.
- Cai thien kha nang nhan dien cac class nho.

## 6. Han che can trinh bay trung thuc

BLIP-GAN la ban cai tien trong project, nhung can trinh bay dung muc:

1. Code van ke thua nhieu ten class cua ACFD-GAN, vi vay khong nen noi BLIP-GAN la mot paper/model hoan toan tach roi ACFD-GAN.
2. Weak mask khong phai lesion annotation chuan y khoa, nen co the bi nhieu hoac thieu lesion.
3. Lesion inpainting giup giu nen, nhung neu mask sai thi vung sinh anh cung co the sai.
4. FID/MSE chi danh gia mot phan chat luong anh; tac dong cuoi cung van can kiem tra qua classifier DR.

## 7. Cach dien dat trong bao cao

Co the trinh bay BLIP-GAN nhu sau:

> BLIP-GAN la phien ban tai cau truc va mo rong tu ACFD-GAN, duoc thiet ke cho bai toan tong hop anh vong mac trong boi canh du lieu DR mat can bang. Mo hinh ke thua cac thanh phan nen tang cua ACFD-GAN nhu WAE latent conditioning, LRDB, ACFF, AMM va PatchGAN discriminator. Diem cai tien chinh cua BLIP-GAN la bo sung weak-mask soft fusion, co che lesion-region inpainting va background preservation loss, giup Generator tap trung tong hop cac vung ton thuong trong khi bao toan nen giai phau cua anh vong mac. Nho do, anh tong hop co xu huong on dinh hon ve cau truc nen, giam artifact va phu hop hon cho muc tieu tang du lieu lop hiem trong phan loai benh vong mac dai thao duong.

## 8. Ket luan

ACFD-GAN la nen tang GAN co dieu kien manh cho tong hop anh DR, voi cac module LRDB, ACFF, AMM va WAE latent. BLIP-GAN trong project nay khong thay the hoan toan ACFD-GAN, ma mo rong ACFD-GAN theo huong thuc dung hon cho du lieu fundus: tao weak mask tu anh that, sinh anh tap trung vao vung lesion, bao toan background va danh gia FID on dinh hon bang multi-seed/post-hoc evaluation.

Noi ngan gon, ACFD-GAN manh o kien truc sinh anh co dieu kien; BLIP-GAN manh hon o kha nang kiem soat vung sinh anh va giam artifact nen, nen phu hop hon voi muc tieu tong hop du lieu hiem cho bai toan DR classification.
