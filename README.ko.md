# VLM 검색 강건성 재현 저장소

이 저장소는 IEEE Access 논문 **“Bidirectional Retrieval Robustness of Four
Vision-Language Checkpoints Under Natural Distortions and Gallery-Contamination
Stress Tests”**의 코드, 설정, rank 체크포인트와 최종 결과표를 공개하기
위한 패키지입니다.

공개 저장소 주소는 <https://github.com/cmony/vlm-retrieval-robustness>입니다.

## 포함 범위

- CLIP ViT-B/32, CLIP ViT-B/16, BLIP base, SigLIP base 평가 코드
- COCO 2017 validation 전체 5,000장 자연 왜곡 설정
- 반복별 metric-first 집계와 조건부 S→F 전이율
- 이미지 클러스터 부트스트랩과 30개 checkpoint contrast
- 자연 왜곡 rank NPZ 140개와 최종 분석 CSV
- RoCOCO 기반 gallery-contamination 결과와 실행 코드
- 논문 Figure 1–5 재생성 코드와 PDF
- 단위 테스트와 공개 패키지 검증 스크립트

COCO 이미지, 모델 가중치와 약 1.2GB의 RoCOCO 임베딩 캐시는 포함하지
않습니다. 원 제공처에서 내려받거나 실행 중 다시 생성하도록 구성했습니다.

## 로컬 검증

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

검증 성공 시 aggregate 152행, 반복별 metric 280행, bootstrap 720행,
checkpoint contrast 30행, 보정 후 유의한 contrast 11개, RoCOCO 20행,
rank NPZ 140개, per-image 720,000행, Figure PDF 5개가 확인됩니다.

자연 왜곡 재분석, GPU 전체 재실행, RoCOCO와 그림 재생성 명령은
[REPRODUCIBILITY.md](REPRODUCIBILITY.md)에 정리했습니다.

## 공개 및 보존 단계

1. 저장소 루트에 이 패키지의 전체 내용을 업로드
2. 기관 정책을 확인한 뒤 라이선스 선택—원저작 코드에는 MIT 권장
3. GitHub Release `v1.0.0` 생성
4. 가능하면 Zenodo를 연결해 버전 고정 DOI 발급
5. DOI가 발급되면 논문과 `CITATION.cff`에 추가
