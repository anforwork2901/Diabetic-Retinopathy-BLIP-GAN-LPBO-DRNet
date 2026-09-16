# Uu nhuoc diem cua ACFD-GAN va huong cai tien trong BLIP-GAN

Tai lieu nay trinh bay ro cac diem manh, han che cua ACFD-GAN va cach BLIP-GAN trong project cai tien dua tren nhung han che do. Noi dung co the dung cho phan "dong gop de tai", "co so de xuat mo hinh" hoac phan tra loi phan bien khi bao ve.

## 1. ACFD-GAN la gi?

ACFD-GAN la viet tat cua Adaptive Cross-layer Fusion and Dense Generative Adversarial Network. Day la mo hinh GAN co dieu kien duoc de xuat de tong hop anh vong mac cho bai toan phan loai benh vong mac dai thao duong.

ACFD-GAN tap trung vao ba muc tieu:

- Tao anh tong hop co chat luong cao.
- Giu lai cac chi tiet lesion nho trong anh vong mac.
- Giam mat can bang lop bang cach sinh them anh cho cac lop benh it mau.

Kien truc ACFD-GAN gom cac thanh phan chinh:

- WAE de hoc latent representation cua anh that.
- LRDB de trich xuat dac trung cuc bo nhe.
- ACFF de hop nhat feature encoder-decoder bang attention.
- AMM de dieu bien feature bang latent vector va mask.
- PatchGAN discriminator de danh gia anh that/gia theo patch cuc bo.
- FID/WMA-FID de chon checkpoint sinh anh tot.

## 2. Uu diem cua ACFD-GAN

### 2.1 Giai quyet bai toan mat can bang du lieu

Trong cac bo du lieu DR nhu APTOS va Messidor, cac lop nang nhu Severe va Proliferative DR thuong co so luong anh rat it. ACFD-GAN giai quyet van de nay bang cach sinh them anh tong hop cho cac lop thieu mau.

Diem manh:

- Giam su phu thuoc vao data augmentation truyen thong.
- Tao them mau moi thay vi chi xoay/lat/doi mau anh cu.
- Ho tro classifier hoc duoc dac trung cua cac lop hiem.

### 2.2 Co dieu kien theo cau truc lesion/mask

ACFD-GAN khong sinh anh hoan toan tu noise ma co su dieu kien tu mask va latent vector. Dieu nay giup mo hinh co kha nang kiem soat noi dung anh sinh.

Diem manh:

- Anh sinh ra co lien quan den cau truc vong mac.
- Vung ton thuong co dinh huong ro hon.
- Giam kha nang sinh anh vo nghia so voi GAN khong dieu kien.

### 2.3 LRDB giu chi tiet nho voi chi phi tinh toan thap

LRDB giup trich xuat dac trung cuc bo va giu lai nhung chi tiet nho nhu vi phinh mach, xuat huyet va tiet cung.

Diem manh:

- Nhe hon residual dense block day du.
- Van giu duoc ket noi residual va dense-style.
- Phu hop voi anh y khoa co nhieu lesion nho.

### 2.4 ACFF giup hop nhat feature nhieu tang

ACFF ket hop feature shallow va deep bang channel attention va spatial attention.

Diem manh:

- Feature shallow giu chi tiet canh/bien.
- Feature deep giu thong tin ngu nghia.
- Attention giup uu tien vung lien quan den lesion.
- Giam mat chi tiet khi decoder phuc hoi anh.

### 2.5 AMM tang tinh da dang va dieu khien qua trinh sinh anh

AMM dung WAE latent va random noise de dieu bien feature.

Diem manh:

- Tang tinh da dang cua anh sinh.
- Giu cau truc tong quat tot hon.
- Giam mode collapse so voi generator chi dung noise.

### 2.6 Co tieu chi chon mo hinh bang FID/WMA-FID

ACFD-GAN dung FID va WMA-FID de chon checkpoint sinh anh.

Diem manh:

- Giam viec chon checkpoint dua tren cam tinh.
- FID danh gia phan phoi feature cua anh that va anh sinh.
- WMA-FID lam muot dao dong FID theo epoch.

## 3. Nhuoc diem cua ACFD-GAN

Mac du ACFD-GAN co kien truc manh, khi ap dung vao project thuc te van co mot so han che.

### 3.1 Phu thuoc vao chat luong mask/dieu kien dau vao

ACFD-GAN can thong tin dieu kien nhu mask hoac structural map de huong dan generator. Neu mask khong chinh xac, generator co the sinh lesion sai vi tri hoac bo sot cac vung quan trong.

Han che:

- Neu khong co annotation lesion chuan, viec tao mask la kho.
- Mask thieu lesion se lam generator khong hoc duoc chi tiet benh.
- Mask nhieu co the lam generator sinh artifact.

BLIP-GAN cai tien:

- Su dung weak-mask soft fusion tu anh that.
- Ket hop Frangi vesselness, Canny, bright lesion cue va dark lesion cue.
- Khong can annotation lesion thu cong.
- Giu lai ca tin hieu yeu cua mach mau va ton thuong nho.

### 3.2 Generator co the lam thay doi qua nhieu vung nen

ACFD-GAN sinh anh theo huong toan cuc. Khi generator sinh lai ca anh, cac vung khong lien quan den benh nhu nen vong mac, mach mau lon, optic disc hoac vien anh co the bi bien dang.

Han che:

- Anh co the nhin dep nhung sai cau truc giai phau.
- Classifier co the hoc artifact thay vi lesion that.
- FID co the tot nhung anh khong that su phu hop ve mat y khoa.

BLIP-GAN cai tien:

- Them co che lesion-region inpainting.
- Chi pha tron anh sinh vao vung lesion-heavy.
- Vung background duoc giu tu anh goc.
- Giam artifact o nen va giu cau truc fundus on dinh hon.

### 3.3 Loss goc chua ep bao toan nen anh

ACFD-GAN dung adversarial loss, L1 loss va perceptual loss. Cac loss nay giup anh sinh gan anh that, nhung khong truc tiep ep generator giu nguyen vung background khong lien quan den lesion.

Han che:

- Generator co the thay doi mau nen hoac texture nen.
- Vung khong benh van co the bi bien doi.
- Anh tong hop co the tao nhieu pattern gia lam nhieu classifier.

BLIP-GAN cai tien:

- Them background preservation loss.
- Tinh L1 o vung background bang trong so `1 - lesion_alpha`.
- Ep generator chi thay doi manh o vung lesion.
- Giu vung nen on dinh va giam bien dang khong can thiet.

### 3.4 ACFD-GAN van co nguy co sinh anh "tot ve FID" nhung chua chac tot cho classifier

FID do khoang cach feature distribution giua anh that va anh sinh. Tuy nhien FID khong dam bao tat ca lesion duoc sinh dung theo y nghia lam sang.

Han che:

- FID thap khong dong nghia voi label-consistency hoan hao.
- Anh co the gan phan phoi that nhung lesion khong ro.
- Mot so anh co the dep ve thi giac nhung khong giup classifier.

BLIP-GAN cai tien:

- Sinh anh dua tren lesion mask de tang lien ket voi vung benh.
- Dung post-hoc FID voi nhieu seed de giam phu thuoc vao mot lan sample.
- Co candidate generation/filtering de chon mau on dinh hon.
- Kiem tra anh sinh qua pipeline classification sau do.

### 3.5 Sinh toan anh de gay nhiem khi du lieu that co nhieu bien thien

Anh fundus co nhieu bien thien ve camera, do sang, mau nen, vung crop, kich thuoc optic disc va chat luong chup. Neu generator hoc toan anh, no phai hoc dong thoi ca background, vessel, optic disc va lesion.

Han che:

- Bai toan sinh anh tro nen kho hon.
- Generator de hoc cac dac trung nen hon la lesion.
- Cac class hiem co it mau nen generator de overfit.

BLIP-GAN cai tien:

- Tach vai tro background va lesion.
- Background duoc lay lai tu anh that.
- Generator tap trung vao vung co lesion.
- Phu hop hon voi bai toan tang du lieu hiem, vi muc tieu chinh la bo sung lesion pattern cho class thieu.

### 3.6 ACFD-GAN can cau hinh on dinh de tranh dao dong GAN

Training GAN de dao dong, dac biet voi du lieu y khoa it mau. ACFD-GAN dung WMA-FID de giam dao dong trong viec chon checkpoint, nhung qua trinh train van co the nhay cam voi seed, learning rate va epoch.

Han che:

- FID co the dao dong theo epoch.
- Mot checkpoint co FID tot co the do seed/candidate may man.
- Training tren class it mau co nguy co bat on.

BLIP-GAN cai tien:

- Them post-hoc FID nhieu seed.
- Co WMA-FID va FID history.
- Co gradient clipping cho generator.
- Co candidate generation de giam rui ro sample xau.
- Co preview real/mask/fake de kiem tra truc quan.

## 4. Bang tom tat: Nhuoc diem ACFD-GAN va cai tien BLIP-GAN

| Nhuoc diem cua ACFD-GAN | Huong cai tien trong BLIP-GAN | Tac dung |
|---|---|---|
| Phu thuoc vao mask chat luong cao | Weak-mask soft fusion tu anh that | Khong can annotation lesion thu cong |
| De thay doi ca vung background | Lesion-region inpainting | Giu nen vong mac on dinh |
| Loss chua ep giu background | Background preservation loss | Giam artifact va bien dang mau nen |
| FID tot chua chac anh tot ve lesion | Lesion-aware mask + post-hoc FID | Tang tinh lien quan voi vung benh |
| Generator phai hoc ca toan anh | Tap trung sinh vung lesion-heavy | Giam do kho cua bai toan sinh anh |
| Dao dong khi training GAN | WMA-FID, multi-seed FID, gradient clipping | Chon checkpoint on dinh hon |
| De hoc artifact cua du lieu it mau | Giu base image va chi thay doi vung benh | Tang do tin cay y khoa cua anh sinh |

## 5. Diem manh rieng cua BLIP-GAN

### 5.1 Lesion-aware generation

BLIP-GAN khong xem moi pixel nhu nhau. Mo hinh uu tien vung lesion thong qua alpha mask. Dieu nay phu hop voi DR vi thong tin quyet dinh grade nam o lesion.

### 5.2 Background-preserving synthesis

BLIP-GAN giu lai vung nen that, giup anh tong hop it bi bien dang giai phau. Day la diem quan trong vi trong anh fundus, nhieu artifact nho co the lam classifier hoc sai.

### 5.3 Khong can mask y khoa thu cong

Weak mask duoc tao tu anh that bang xu ly anh. Dieu nay lam pipeline de tai kha thi hon vi khong can chuyen gia gan nhan tung lesion.

### 5.4 Phu hop voi tang du lieu lop hiem

BLIP-GAN co the sinh anh theo tung grade, dac biet la cac lop 3 va 4. Cac anh nay duoc dung de can bang du lieu va ho tro classifier hoc tot hon tren lop hiem.

### 5.5 Giu duoc nen tang manh cua ACFD-GAN

BLIP-GAN khong phu nhan ACFD-GAN ma ke thua:

- WAE latent.
- LRDB.
- ACFF.
- AMM.
- PatchGAN.
- Perceptual loss.
- FID/WMA-FID.

Sau do bo sung cac co che thuc nghiem de giam artifact va tang tinh lesion-aware.

## 6. Cach trinh bay khi bao ve

Neu hoi: "ACFD-GAN da manh roi, tai sao can BLIP-GAN?"

Co the tra loi:

> ACFD-GAN da co nen tang rat tot voi LRDB, ACFF, AMM va WAE latent. Tuy nhien, khi ap dung vao anh vong mac thuc te, generator sinh toan anh co the lam bien dang nhung vung khong lien quan den benh nhu background, optic disc va mach mau. Ngoai ra, neu khong co mask lesion chuan, chat luong dieu kien dau vao co the anh huong den anh sinh. Vi vay, em cai tien thanh BLIP-GAN bang cach tao weak mask tu anh that, chi sinh/pha tron manh o vung lesion-heavy va them background preservation loss de giu nen vong mac on dinh. Cach nay giup anh tong hop tap trung vao vung benh hon, giam artifact nen va phu hop hon voi muc tieu tang du lieu lop hiem cho bai toan DR.

Neu hoi: "BLIP-GAN khac ACFD-GAN o dau?"

Co the tra loi:

> BLIP-GAN van ke thua backbone sinh anh cua ACFD-GAN, nhung khac o pipeline dieu kien va cach tong hop anh dau ra. Thay vi chi dua vao mask/latent va sinh lai toan anh, BLIP-GAN tao weak mask lesion-aware tu anh that, dung alpha lesion de inpaint vung ton thuong, dong thoi giu lai background tu anh goc. Ngoai ra, BLIP-GAN them background preservation loss va danh gia FID voi nhieu seed de tang do on dinh.

Neu hoi: "Cai tien nay co y nghia gi voi phan loai DR?"

Co the tra loi:

> Trong DR, cac chi tiet quyet dinh grade thuong la lesion nho nhu microaneurysm, hemorrhage, exudate. Neu GAN sinh toan anh va tao artifact o background, classifier co the hoc sai. BLIP-GAN tap trung vao vung lesion va giu background on dinh, nen anh sinh co kha nang bo sung thong tin benh tot hon cho cac lop hiem, thay vi chi tao bien the hinh anh khong lien quan.

## 7. Ket luan

ACFD-GAN co uu diem lon ve kien truc sinh anh co dieu kien, dac biet la LRDB, ACFF, AMM va WAE latent. Tuy nhien, han che cua no khi ap dung vao du lieu DR thuc te la phu thuoc vao mask, co nguy co thay doi background va chua truc tiep ep generator tap trung vao vung lesion.

BLIP-GAN cai tien cac diem do bang weak-mask soft fusion, lesion-region inpainting, background preservation loss va danh gia FID on dinh hon. Vi vay, BLIP-GAN phu hop hon voi muc tieu cua do an: tong hop du lieu hiem co tinh lesion-aware va ho tro phan loai benh vong mac dai thao duong.
