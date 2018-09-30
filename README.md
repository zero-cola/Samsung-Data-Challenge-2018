# Samsung-Data-Challenge-2018
주제: 과거 교통사고 데이터를 분석하여 미래 교통사고 예측하기 (https://research.samsung.com/aichallenge/data)
발표자료: [slideshare](https://www.slideshare.net/KyusooHan/missing-data-prediction-using-nmdae)

# Data
예측해야 하는 데이터는 다음과 같은 수치형 변수와 범주형 변수가 주어진다.

| 주야  | 요일 | 사망자수 | 사상자수  | ... | 발생지시도  | 발생지시군구 | 당사자 1당   | 당사자 2당 |
|:----:|:----:|:--------:|:--------:|:---:|:----------:|:------------:|:----------:|:----------:|
| *NaN* |  화  |   *NaN* |     3    | ... |    *NaN*   |    마포구    |   *NaN*     |   자전거   |

모델은 가용 데이터를 사용하여 다음과 같이 복수의 NaN 데이터를 예측한다.  

| 주야  | 요일 | 사망자수 | 사상자수  | ... | 발생지시도  | 발생지시군구  | 당사자 1당 | 당사자 2당  |
|:----:|:----:|:--------:|:--------:|:---:|:----------:|:------------:|:----------:|:----------:|
|*주간* |  화  |     *2* |     3    | ... |    *서울*   |    마포구    |   *승용차*  |   자전거   |

# Model
NaN Mask Denoising Auto Encoder

# Result
1st Prize

# Authors
**Yongwoo Cho(조용우)**
- https://nomorecoke.github.io
- https://github.com/nomorecoke

**Gyusu Han(한규수)**
- https://gyusu.github.io
- https://github.com/gyusu
