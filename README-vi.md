# CVR for Very Large Regression Problems

*Đọc bằng ngôn ngữ khác: [English](README.md)*

Kho lưu trữ này chứa mã nguồn triển khai chính thức và các Jupyter Notebook để đánh giá mô hình Core Vector Regression và Core Vector Machine, được tối ưu hóa đặc biệt cho các bài toán quy mô cực lớn.

## Tham khảo

> Ivor W. Tsang, James T. Kwok, and Kimo T. Lai. 2005. Core Vector Regression for very large regression problems. In Proceedings of the 22nd international conference on Machine learning (ICML '05). Association for Computing Machinery, New York, NY, USA, 912–919. [https://doi.org/10.1145/1102351.1102466](https://doi.org/10.1145/1102351.1102466)

## Cấu trúc thư mục

- **`implement/`**: Chứa mã nguồn C++ cốt lõi của các mô hình đã được tối ưu hóa.
  - `cvm_train.cpp`: Triển khai mô hình Core Vector Machine cho các bài toán phân loại.
  - `cvr_train.cpp`: Triển khai mô hình Core Vector Regression cho các bài toán hồi quy.

  *Lưu ý: Cả hai triển khai đều sử dụng thuật toán SMO (Sequential Minimal Optimization) và được tối ưu hóa mạnh mẽ bằng các tập lệnh AVX2/BMI để đạt hiệu suất tối đa. Đầu vào yêu cầu dữ liệu chuẩn theo định dạng LIBSVM.*

- **`evaluation/`**: Chứa các Jupyter Notebook dùng để đánh giá hiệu năng của các mô hình.
  - `large-benchmark-data-sets/`:
    - `cvr-full.ipynb`: Notebook tổng hợp để đánh giá mô hình trên toàn bộ các tập dữ liệu quy mô lớn.
    - `parts/`: Chứa các phiên bản chia nhỏ của `cvr-full.ipynb` chạy trên từng thiết lập cụ thể (để tránh quá tải bộ nhớ hoặc giới hạn thời gian chạy trên cloud).

      **Quy ước tên mô hình so sánh**:
      - **CVR**: L2-SVR (Core Vector Regression)
      - **LIBSVM**: L1-SVR (LIBSVM)
      - **SVMlight**: L1-SVR (SVM^light)
      - **CVM**: L2-SVM (Core Vector Machine)
      - **nu-SVR**: $\nu$-SVR (LIBSVM)

      **Các Notebook Benchmark cho bài toán Hồi quy**

      | Notebook File | Tập dữ liệu | Mô hình | Các mốc (%) |
      | --- | --- | --- | --- |
      | `01_standard_benchmarks.ipynb` | `census_housing`, `computer_activity`, `elevators`, `pole_telecomm` | CVR, LIBSVM, SVMlight | 10, 30, 50, 90 |
      | `02_novel_datasets.ipynb` | `superconductivity`, `bike_sharing` | CVR, LIBSVM, SVMlight | 10, 30, 50, 90 |
      | `03_friedman_cvr_vs_libsvm.ipynb` | `friedman` | CVR, LIBSVM | 5, 10, 20, 35, 50, 70, 90 |
      | `04_friedman_all_models.ipynb` | `friedman` | CVR, LIBSVM, SVMlight | 5, 10, 20, 35, 50 |

      **Các Notebook Benchmark cho bài toán Phân loại**

      | Notebook File | Tập dữ liệu | Mô hình | Các mốc (%) |
      | --- | --- | --- | --- |
      | `05_covertype_nusvr.ipynb` | `covertype` | nu-SVR | 1, 2, 5, 10, 20 |
      | `06_covertype_cvm_cvr.ipynb` | `covertype` | CVM, CVR | 1, 2, 5, 10, 20, 30, 50, 70, 90 |

  - `surface-modeling/`: Notebooks dùng để đánh giá trên các tác vụ tái tạo bề mặt 3D sử dụng mô hình Core Vector Regression multi-scale trên dữ liệu point cloud.
    - `01_stanford_bunny_multiscale.ipynb`: Mô hình hóa bề mặt cho tập dữ liệu Stanford Bunny.
    - `02_stanford_dragon_multiscale.ipynb`: Mô hình hóa bề mặt cho tập dữ liệu Stanford Dragon.

- **`ablation/`**: Chứa các Jupyter Notebook thực hiện ablation study. Các notebook này phân tích chi tiết tác động của từng siêu tham số (như epsilon, mu, gamma) và các thành phần (như cấu trúc nhân kernel, các bước tối ưu SMO) lên kích thước mô hình, thời gian hội tụ và độ chính xác.
  - `01_ablation_hyperparams_components_benchmark.ipynb`: Phân tích các siêu tham số và thành phần trên các tập dữ liệu benchmark tiêu chuẩn.
  - `02_ablation_hyperparams_components_surface.ipynb`: Phân tích các siêu tham số và thành phần chuyên biệt trên tác vụ mô hình hóa bề mặt 3D.

## Thông tin Datasets

### Các tập dữ liệu Benchmark

| Tên Dataset | Loại bài toán | Số đặc trưng | Số lượng mẫu |
| --- | --- | --- | --- |
| `census_housing` | Hồi quy | 121 | 22.732 |
| `computer_activity` | Hồi quy | 21 | 8.192 |
| `elevators` | Hồi quy | 18 | 16.599 |
| `friedman` | Hồi quy | 10 | 100.000 |
| `pole_telecomm` | Hồi quy | 48 | 15.000 |
| `superconduct` | Hồi quy | 81 | 21.263 |
| `bike_sharing` | Hồi quy | 12 | 17.379 |
| `covertype` | Phân loại | 54 | 581.012 |

### Các tập dữ liệu Surface Modeling & Tham số chạy

| Tên Dataset | Loại Dữ liệu | Kernel | Sigmas | C | mu | epsilon |
| --- | --- | --- | --- | --- | --- | --- |
| `Stanford Bunny` | 3D Point Cloud (10,000 pts) | Laplacian | [0.2, 0.1, 0.05] | 1.0 | 1e-4 | 1e-4 |
| `Stanford Dragon` | 3D Point Cloud (30,000 pts) | Laplacian | [0.1, 0.05, 0.02] | 10.0 | 1e-4 | 1e-5 |

*Lưu ý: Tác vụ mô hình hóa bề mặt sử dụng Laplacian Kernel (`exp(-gamma * ||x-y||)`) và được huấn luyện lặp qua nhiều tỷ lệ (với `gamma = 1/sigma`) để mô phỏng từ hình dáng thô sơ, chi tiết trung bình đến các chi tiết sắc nét.*

## Cài đặt & Biên dịch (Môi trường Local)

Các file mã nguồn C++ trong thư mục `implement/` có thể được biên dịch bằng `g++` với các cờ tối ưu hóa. Ví dụ:

```bash
g++ -O3 -std=c++11 implement/cvm_train.cpp -o cvm_train
g++ -O3 -std=c++11 implement/cvr_train.cpp -o cvr_train
```

Các file thực thi sau khi biên dịch yêu cầu các tham số truyền vào như đường dẫn tới file dữ liệu huấn luyện (định dạng LIBSVM), các siêu tham số (C, gamma, epsilon...) và đường dẫn file đầu ra. Vui lòng tham khảo trực tiếp mã nguồn để biết thứ tự tham số chính xác.

## Chạy Notebooks trên Kaggle/Colab

Các notebook trong thư mục `evaluation/` và `ablation/` được thiết kế riêng để chạy trên các nền tảng đám mây như Kaggle và Google Colab. **Các notebook này sẽ không thể chạy trực tiếp trên máy tính local** vì chúng chứa các đường dẫn tuyệt đối dành riêng cho môi trường cloud và các lệnh hệ thống để cài đặt/biên dịch các thư viện cơ sở (`libsvm`, `SVM^light`, `libCVM`).

Để sử dụng các notebook này:

1. **Kaggle**:
   - Tạo một notebook mới trên Kaggle.
   - Tải file notebook từ kho lưu trữ này lên bằng tùy chọn `File -> Import Notebook`.
   - Chạy các cell. Notebook sẽ tự động tải dữ liệu, biên dịch và chạy thí nghiệm.
2. **Google Colab**:
   - Truy cập [Google Colab](https://colab.research.google.com/).
   - Chọn `Upload` và tải lên file notebook bạn muốn chạy.
   - Chạy các cell để thực thi thí nghiệm.

> [!IMPORTANT]  
> **Lưu ý về đường dẫn:** Thư mục làm việc mặc định của Kaggle là `/kaggle/working/`, trong khi của Google Colab là `/content/`. Do các notebook được export từ các môi trường khác nhau, một số cell có thể chứa đường dẫn cứng:
>
> - Nếu bạn đang chạy trên **Colab** mà thấy đường dẫn `/kaggle/working/` trong mã nguồn, hãy tìm và đổi nó thành `/content/`.
> - Ngược lại, nếu chạy trên **Kaggle** mà thấy đường dẫn `/content/`, hãy tìm và đổi nó thành `/kaggle/working/`.
