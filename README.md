# ColabNAS cho bài toán phân lớp ảnh với tài nguyên hạn chế

Source code khóa luận tốt nghiệp của nhóm sinh viên trường Đại học Khoa học Tự nhiên, ĐHQG-HCM, xây dựng dựa trên ColabNAS để tìm kiến trúc CNN nhẹ, phù hợp triển khai trên vi điều khiển (MCU) tài nguyên hạn chế cho bài toán phân lớp ảnh.

## Thông tin khóa luận

- **Trường:** Đại học Khoa học Tự nhiên, ĐHQG-HCM — Khoa Công nghệ Thông tin
- **Đề tài:** Mô hình CNN nhẹ cho bài toán phân lớp ảnh với tài nguyên hạn chế
- **GVHD:** TS. Bùi Duy Đăng — Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM
- **Nhóm sinh viên:**
  - 22127212 - Uông Minh Nguyên Khôi — Chuyên ngành: Khoa học máy tính
  - 22127481 — Nguyễn Thanh Nhàn — Chuyên ngành: Công nghệ Thông tin

## Giới thiệu

Việc triển khai mô hình trên các thiết bị có tài nguyên phần cứng hạn chế mang lại nhiều ý nghĩa thực tiễn to lớn, đặc biệt trong việc tối ưu hóa chi phí và tiết kiệm năng lượng tiêu thụ. Trong thời gian gần đây, hướng tiếp cận sử dụng mô hình mạng nơ-ron tích chập (CNN) nhẹ mang lại nhiều kết quả tích cực. Trong đó, nổi lên nhóm phương pháp neural architecture search (NAS) có thể tìm kiếm kiến trúc mạng thỏa mãn các cấu hình phần cứng cho trước. Do đó, đây là hướng tiếp cận mà khóa luận chọn để tìm hiểu.

ColabNAS là một phương pháp thuộc nhóm NAS, tìm kiếm kiến trúc mạng bằng thuật toán không dùng đến đạo hàm, theo nguyên lí "Dao cạo Occam" để giữ cho mô hình đơn giản. Nhờ đó, toàn bộ quá trình huấn luyện có thể được thực hiện trên các tài nguyên GPU miễn phí trên GoogleColab, Kaggle,... Do đó phương pháp này được đặt tên là ColabNAS.

Ngoài tái hiện lại các thí nghiệm được công bố trong bài báo gốc, khóa luận cũng thực hiện thêm một vài đề xuất cải tiến đơn giản để khắc phục những vấn đề về tính "ngẫu nhiên" trong cài đặt gốc của ColabNAS.

## ⚠️ Nguồn gốc & mức độ kế thừa từ bài báo gốc

Source code này được xây dựng **dựa trên** ColabNAS (Garavagno et al., 2024) — phần lõi thuật toán
(`class ColabNAS`, hàm `search`, `explore_num_cells`) là tái hiện/kế thừa trực tiếp từ [repo gốc](https://github.com/AndreaMattiaGaravagno/ColabNAS). Phần lớn khối lượng "code mới" trong khóa luận là hiện thực lại các thí nghiệm mà bài báo có mô tả bằng lời/bảng số liệu nhưng notebook gốc chưa từng viết code (xem chi tiết đối chiếu từng dòng trong **[MODIFIED.md](MODIFIED.md)**).

👉 **Đọc [MODIFIED.md](MODIFIED.md) trước khi đánh giá phần "đóng góp mới"** — file này tách rõ đâu là phần mã hoá lại ý tưởng có sẵn của bài báo, đâu là 2 ý tưởng thực sự mới của nhóm (hiện đang để dạng code comment, chưa bật mặc định — xem mục "Giới hạn hiện tại" bên dưới).

### Giấy phép & trích dẫn

Repo gốc ColabNAS được phát hành theo **MIT License** (Copyright (c) 2023 AndreaMattiaGaravagno).
Repo khóa luận này giữ nguyên MIT License — xem file [LICENSE](LICENSE), trong đó giữ nguyên copyright notice của tác giả gốc theo đúng điều khoản MIT, cộng thêm copyright notice của nhóm cho phần code mới bổ sung.

Nếu sử dụng lại ý tưởng/thuật toán trong repo này, vui lòng trích dẫn bài báo gốc:

```bibtex
@article{GARAVAGNO2024152,
    title = {ColabNAS: Obtaining lightweight task-specific convolutional neural networks following Occam's razor},
    author = {Andrea Mattia Garavagno and Daniele Leonardis and Antonio Frisoli},
    journal = {Future Generation Computer Systems},
    volume = {152},
    pages = {152-159},
    year = {2024},
    issn = {0167-739X},
    doi = {https://doi.org/10.1016/j.future.2023.11.003}
}
```

## Cấu trúc repository

| File | Mô tả |
|---|---|
| `T826_KL_KHMT24_SourceCode.ipynb` | Notebook chính của khóa luận — toàn bộ code thí nghiệm |
| `ORGINAL_ColabNAS.ipynb` | Notebook gốc đi kèm bài báo |
| `MODIFIED.md` | Đối chiếu chi tiết giữa notebook khóa luận và bài báo/notebook gốc |
| `requirements.txt` | Các thư viện Python cốt lõi cần cho notebook |
| `LICENSE` | MIT License — giữ nguyên copyright của tác giả gốc, cộng thêm copyright của nhóm cho phần code mới |
| `README.md` | File này |

## Bộ dữ liệu sử dụng

| Dataset | Bài toán | Nguồn / Trích dẫn |
|---|---|---|
| Melanoma Skin Cancer | Phân loại benign/malignant | https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images |
| Flowers-4 | Phân loại 4 loại hoa (tập con Flowers) | Bộ gốc ([l3llff/flowers](https://www.kaggle.com/datasets/l3llff/flowers)) hiện đã bị gỡ khỏi Kaggle. Khóa luận dùng bản sao do tác giả A. M. Garavagno tải về trước đó, lưu trữ tại [Google Drive](https://drive.google.com/file/d/18NHZUFfDyPzjsTTcTXFEtZQNC5Fesnxl/view?usp=drive_link) |
| Animals-3 | Phân loại 3 loài động vật (tập con Animals-10) | https://www.kaggle.com/datasets/alessiocorrado99/animals10 |
| MNIST | Phân loại chữ số viết tay | LeCun, Y., Cortes, C., and Burges, C. J. *MNIST handwritten digit database*. AT&T Labs, 2010. http://yann.lecun.com/exdb/mnist (cũng có trên Kaggle: [hojjatk/mnist-dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)) |
| Visual Wake Words (VWW) | Phát hiện có/không có người trong ảnh | Chowdhery, A., Warden, P., Shlens, J., Howard, A., and Rhodes, R. *Visual Wake Words Dataset*, 2019. arXiv: [1906.05721](https://arxiv.org/abs/1906.05721) [cs.CV] |

*Lưu ý: trong `data_dirs` của notebook, một số dataset trên được mount từ bản re-upload trên tài khoản Kaggle cá nhân của nhóm (`karosvn`, `tommerfrancis`) để đảm bảo notebook chạy ổn định trên Kaggle; nguồn/trích dẫn học thuật chính thức vẫn là các nguồn liệt kê ở trên.*

## Môi trường & cách chạy

Notebook được thiết kế để chạy trên **Kaggle Notebook**, không chạy được ở máy local, vì:

- Dataset được gắn vào dưới dạng Kaggle Dataset input (`/kaggle/input/datasets/...`), output ghi vào `/kaggle/working/`.
- Việc đo RAM/Flash đỉnh (peak) của mô hình cần chạy binary `stm32tflm` — hiện notebook dùng bản đã biên dịch sẵn cho **Linux**, upload lên Kaggle Dataset cá nhân (`karosvn/stm32tflm-linux`).

Các bước chạy:
1. Import `T826_KL_KHMT24_SourceCode.ipynb` vào Kaggle, attach đủ các Dataset input ở bảng trên.
2. Bật GPU accelerator T4 x 2 (Kaggle).
3. Chạy tuần tự từ đầu; phần "Cấp quyền thực thi cho `stm32tflm`" phải chạy trước các thí nghiệm.
4. (Tuỳ chọn, để build lại `stm32tflm` từ đầu thay vì dùng bản đã biên dịch sẵn): tải [ST X-CUBE-AI](https://www.st.com/en/embedded-software/x-cube-ai.html#get-software), cài vào STM32CubeMX, dùng công cụ export/analyze để lấy binary `stm32tflm` tương ứng target MCU, rồi `chmod +x` trước khi chạy trên Linux/Kaggle.


## Kết quả thực nghiệm

### Hardware-aware search trên các phần cứng hạn chế (Thí nghiệm 1)

`ColabNAS` chạy trên 5 dataset, mỗi dataset thử với 3 mục tiêu phần cứng L0/L1/L4
(STM32L010RBT6 / STM32L151UCY6DTR / STM32L412KBU3). Dòng "(paper)" là số liệu bài báo gốc báo cáo,
các dòng L0/L1/L4 còn lại là kết quả nhóm tái tạo.

#### Melanoma Skin Cancer (test set chia sẵn)

| Target | Test accuracy (%) | Search time (hh:mm:ss) | RAM (KiB) | Flash (KiB) | MACC (k) |
|---|---|---|---|---|---|
| L0 (paper) | 86.5 | 00:17:00 | 19.5 | 8.3 | 92 |
| L1 (paper) | 88.7 | 00:19:00 | 22 | 14.4 | 654 |
| L4 (paper) | 90.1 | 00:17:00 | 32.5 | 31.84 | 2075 |
| L0 | 82.1 | 00:02:41 | 20 | 4.6 | 270 |
| L1 | 89.2 | 00:10:50 | 23 | 19.1 | 657 |
| L4 | 89.6 | 00:15:33 | 22.5 | 14.9 | 654 |

#### Flowers-4 (test set tự chia)

| Target | Test accuracy (%) | Search time (hh:mm:ss) | RAM (KiB) | Flash (KiB) | MACC (k) |
|---|---|---|---|---|---|
| L0 (paper) | 79.2 | 00:05:00 | 18.5 | 6.67 | 211 |
| L1 (paper) | 84.4 | 00:08:00 | 21.5 | 10.91 | 633 |
| L4 (paper) | 91.3 | 00:10:00 | 32.5 | 31.91 | 2075 |
| L0 | 77.31 | 00:01:18 | 20 | 4.8 | 270 |
| L1 | 85.86 | 00:03:47 | 21.5 | 11.4 | 633 |
| L4 | 89.54 | 00:05:16 | 32.5 | 32.8 | 2075 |

#### Animals-3 (test set tự chia)

| Target | Test accuracy (%) | Search time (hh:mm:ss) | RAM (KiB) | Flash (KiB) | MACC (k) |
|---|---|---|---|---|---|
| L0 (paper) | 67.9 | 00:09:00 | 19 | 8.03 | 227 |
| L1 (paper) | 75.8 | 00:17:00 | 22.5 | 18.65 | 657 |
| L4 (paper) | 85.2 | 00:09:00 | 33 | 44.86 | 2086 |
| L0 | 55.61 | 00:04:21 | 18.5 | 6.9 | 211 |
| L1 | 69.06 | 00:05:07 | 21.5 | 11.3 | 633 |
| L4 | 79.14 | 00:06:59 | 23 | 19.2 | 657 |

#### MNIST (test set chia sẵn)

| Target | Test accuracy (%) | Search time (hh:mm:ss) | RAM (KiB) | Flash (KiB) | MACC (k) |
|---|---|---|---|---|---|
| L0 (paper) | 88.2 | 01:07:00 | 19.5 | 9.79 | 233 |
| L1 (paper) | 95.6 | 01:41:00 | 22.5 | 18.8 | 657 |
| L4 (paper) | 98 | 01:29:00 | 33 | 45.23 | 2087 |
| L0 | 89.37 | 00:38:30 | 19.5 | 10.3 | 233 |
| L1 | 96.14 | 00:49:16 | 23 | 19.5 | 657 |
| L4 | 97.98 | 00:58:23 | 33.5 | 46.4 | 2087 |

#### Visual Wake Words (test set chia sẵn)

| Target | Test accuracy (%) | Search time (hh:mm:ss) | RAM (KiB) | Flash (KiB) | MACC (k) |
|---|---|---|---|---|---|
| L0 (paper) | 69.4 | 02:11:00 | 19 | 8.02 | 227 |
| L1 (paper) | 74.5 | 03:04:00 | 22.5 | 18.5 | 657 |
| L4 (paper) | 77.8 | 02:47:00 | 33 | 44.9 | 2086 |
| L0 | 69.77 | 01:43:22 | 19.5 | 10.1 | 233 |
| L1 | 73.04 | 01:46:50 | 22.5 | 14.9 | 654 |
| L4 | 76.34 | 01:44:47 | 32 | 21.7 | 1992 |

*RAM/Flash tính bằng KiB (kibibyte); MACC tính bằng k (nghìn phép nhân-cộng) — lưu ý đơn vị MACC ở
đây khác với bảng so sánh Transfer Learning bên dưới (tính bằng MM, triệu).*

### So sánh với Transfer Learning (Thí nghiệm 2)

`ColabNAS` so với `TLModel` (MobileNetV2 backbone đóng băng) trên cùng ràng buộc RAM/Flash/MACC.
Dòng "(paper)" là số liệu bài báo gốc báo cáo, các dòng còn lại là kết quả nhóm tái tạo. **In đậm**
= giá trị tốt hơn giữa TL và ColabNAS trong cùng nhóm so sánh (paper với paper, tái tạo với tái tạo).

#### Melanoma Skin Cancer (test set chia sẵn)

| | Test accuracy (%) | Search time (hh:mm:ss) | RAM (KiB) | Flash (KiB) | MACC (MM) |
|---|---|---|---|---|---|
| TL (paper) | 88 | 00:16:00 | 2523 | 2650 | 300 |
| ColabNAS (paper) | **91.1** | 03:52:00 | **547** | **75.2** | **44** |
| TL | **94.09** | 00:18:06 | 2523 | 2650 | 300 |
| ColabNAS | 90.3 | 02:28:35 | **545.5** | **45.8** | **43** |

#### Animals-3 (test set tự chia)

| | Test accuracy (%) | Search time (hh:mm:ss) | RAM (KiB) | Flash (KiB) | MACC (MM) |
|---|---|---|---|---|---|
| TL (paper) | **99.2** | 00:11:00 | 2523 | 2653 | 300 |
| ColabNAS (paper) | 93.2 | 03:02:00 | **988** | **197.3** | **153** |
| TL | **98.68** | 00:11:58 | 2523 | 2653 | 300 |
| ColabNAS | 92.98 | 02:17:10 | **990** | **254.6** | **153** |

#### Flowers-4 (test set tự chia)

| | Test accuracy (%) | Search time (hh:mm:ss) | RAM (KiB) | Flash (KiB) | MACC (MM) |
|---|---|---|---|---|---|
| TL (paper) | **99.2** | 00:07:00 | 2523 | 2653 | 300 |
| ColabNAS (paper) | 93.8 | 01:20:00 | **546** | **59.7** | **44** |
| TL | **97.19** | 00:06:41 | 2523 | 2653 | 300 |
| ColabNAS | 92.8 | 01:21:40 | **988** | **147.2** | **152** |

*RAM/Flash tính bằng KiB (kibibyte); MACC tính bằng MM (triệu phép nhân-cộng).*

### So sánh SOTA trên VWW (Thí nghiệm 3)

`ColabNAS` so với MCUNet và Micronets (số liệu SOTA lấy từ bài báo gốc) trên dataset Visual Wake
Words (test set chia sẵn), dưới ràng buộc phần cứng STM32F446RE. MCUNet/Micronets không phải kết
quả của quá trình search nên không có cột "Search time".

| | Test accuracy (%) | Search time (hh:mm:ss) | RAM (KiB) | Flash (KiB) | Latency (mS) |
|---|---|---|---|---|---|
| MCUNet | 87.4 | — | 168.5 | 530.52 | 2.16 |
| Micronets | 76.8 | — | 70.5 | 273.81 | 1.15 |
| ColabNAS (paper) | 77.6 | 03:06:00 | 31.5 | 20.83 | 0.432 |
| ColabNAS | 77.86 | 02:16:07 | 53.5 | 30.6 | 0.357 |

*Latency đo bằng `%timeit interpreter.invoke()` sau khi chạy model qua `stm32tflm` (xem mục "Môi
trường & cách chạy" ở trên).*

- Sheet tổng hợp kết quả thực nghiệm (bản merge): [link](https://docs.google.com/spreadsheets/u/2/d/1vhcM7rgDAb9zhxqglKIm8WgOo8VIKDjLWSY92_72KLo/edit?gid=758639362#gid=758639362)
- Notebook Kaggle (bản merge): [link](https://www.kaggle.com/code/karosvn/kltn-nh-n-kh-i-source-code/edit)

## Giới hạn / lưu ý hiện tại

Theo `MODIFIED.md`, khóa luận có 2 ý tưởng thực sự vượt ra ngoài bài báo gốc, nhưng **hiện đang để
dạng code comment, chưa bật mặc định** trong kết quả báo cáo chính thức:

1. Cơ chế fixed-seed/determinism (`FIXED_SEED`, `set_random_seed`, `enable_op_determinism`) để tăng khả năng tái lập kết quả.
2. Hai biến thể thay thế cho `explore_num_cells` ("thử add thêm cell", và "thử duyệt cell top-down").

## Tài liệu khóa luận

- Đề cương: [link](https://www.overleaf.com/8858577417rbxwppmqbtgd#45def1)
- Báo cáo Khóa luận: [link](https://www.overleaf.com/4338641393qqtcgfrzqqkv#39b9f1)
- Slide: [link](https://docs.google.com/presentation/d/1LagNCOUPE4PFQHaIdiw1dvGXU_CxlFIGoXlx16kO11E/edit?usp=sharing)
- Kế hoạch: [link](https://docs.google.com/spreadsheets/d/15bgjC9sBM9WLEpcgA84cZjkK1MEHjC0VrEY7j161dAY/edit?usp=sharing)
- Sheet tổng hợp các paper đã khảo sát: [link](https://docs.google.com/spreadsheets/d/1zpYtDSbJdRINkpCxvo2PxE1RPMMh6ihKPEV9uD7P3Zk/edit?gid=0#gid=0)

## Tham khảo

- Bài báo gốc: Garavagno, A. M., Leonardis, D., & Frisoli, A. (2024). ColabNAS: Obtaining lightweight task-specific convolutional neural networks following Occam's razor. *Future Generation Computer Systems*, 152, 152-159. https://doi.org/10.1016/j.future.2023.11.003
- Repo gốc: https://github.com/AndreaMattiaGaravagno/ColabNAS
- STMicroelectronics X-CUBE-AI: https://www.st.com/en/embedded-software/x-cube-ai.html#get-software
- Chowdhery, A., Warden, P., Shlens, J., Howard, A., and Rhodes, R. (2019). *Visual Wake Words Dataset*. arXiv:1906.05721 [cs.CV]. https://arxiv.org/abs/1906.05721
- LeCun, Y., Cortes, C., and Burges, C. J. (2010). *MNIST handwritten digit database*. AT&T Labs. http://yann.lecun.com/exdb/mnist

## Liên hệ

- Uông Minh Nguyên Khôi — 22127212 — umnkhoi22@clc.fitus.edu.vn — [GitHub](https://github.com/NgKhoi1005)
- Nguyễn Thanh Nhàn — 22127481 — ntnhan224@clc.fitus.edu.vn — [GitHub](https://github.com/nhannguyen0906)
- GVHD: TS. Bùi Duy Đăng — bddang@fit.hcmus.edu.vn
