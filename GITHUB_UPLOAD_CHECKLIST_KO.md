# GitHub 공개 체크리스트

## 권장 저장소 설정

- 저장소 이름: `vlm-retrieval-robustness`
- 설명: `Reproducibility package for bidirectional VLM retrieval robustness under natural distortions and gallery contamination.`
- 공개 범위: 논문 제출 시점에는 `Public` 권장
- GitHub에서 README, `.gitignore`, License 자동 생성: 선택하지 않음
  - 이 폴더에 README와 `.gitignore`가 이미 있습니다.
  - 라이선스는 기관 정책 확인 후 별도로 추가합니다.

## 업로드할 폴더

로컬 `vlm-retrieval-robustness` 폴더의 **내용 전체**를 저장소
루트에 올립니다. 바깥쪽 폴더를 한 번 더 중첩하지 않습니다.

## 명령줄 업로드 예시

아래 명령은 생성된 빈 저장소에 이 패키지를 업로드합니다.

```powershell
Set-Location "D:\내논문\ieee\vlm-retrieval-robustness"
git init
git branch -M main
git add .
git commit -m "Release reproducibility package v1.0.0"
git remote add origin "https://github.com/cmony/vlm-retrieval-robustness.git"
git push -u origin main
```

## URL 생성 후 수정할 곳

1. `CITATION.cff`에 다음 주소가 있는지 확인

```yaml
repository-code: "https://github.com/cmony/vlm-retrieval-robustness"
```

2. 논문 Data Availability에 같은 URL이 삽입되었는지 확인
3. GitHub Releases에서 `v1.0.0` 태그 생성
4. 가능하면 Zenodo와 연결해 DOI를 발급하고 논문과 `CITATION.cff`에 추가

## 공개 직전 검증

```powershell
python scripts\verify_release.py
python -m unittest discover -s tests -v
```

예상 결과는 `RELEASE GATE PASSED`와 34개 테스트 성공입니다.
