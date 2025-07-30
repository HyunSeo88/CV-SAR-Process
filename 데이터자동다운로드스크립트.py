"""
Sentinel‑1 SLC list → 자동 다운로드 스크립트
-------------------------------------------------
1) text_block 에서 SAFE 파일명 추출
2) Catalogue OData API 로 UUID(Id) 조회
3) Access·Refresh 토큰 자동 갱신
4) Range 이어받기 + .part → .zip 완성
"""

import re, requests, time, getpass, pathlib, json
from tqdm import tqdm
import os, time
# ──────────────────────────────────────────
# 0. 계정 정보 (이메일=ID), 비밀번호는 입력
EMAIL = "hyunseo081507@gmail.com"
PW    = getpass.getpass("Copernicus Password: ")

# 1. 텍스트 원본 → SAFE 이름만 추출
text_block = """
S1B_IW_SLC__1SDV_20200703T212335_20200703T212402_022312_02A59D_986B.SAFE
Sensing date
4/7/2020
Cloud coverage
-
Availability
Immediate
Size
7.51 GB

Product details

S1A_IW_SLC__1SDV_20200709T092341_20200709T092410_033376_03DDF1_7AFB.SAFE
Sensing date
9/7/2020
Cloud coverage
-
Availability
Immediate
Size
7.75 GB

Product details

S1B_IW_SLC__1SDV_20200720T213132_20200720T213200_022560_02AD16_11B5.SAFE
Sensing date
21/7/2020
Cloud coverage
-
Availability
Immediate
Size
7.66 GB

Product details

S1A_IW_SLC__1SDV_20200721T092341_20200721T092411_033551_03E34E_961D.SAFE
Sensing date
21/7/2020
Cloud coverage
-
Availability
Immediate
Size
7.75 GB

Product details

S1B_IW_SLC__1SDV_20200801T213133_20200801T213201_022735_02B262_26E9.SAFE
Sensing date
2/8/2020
Cloud coverage
-
Availability
Immediate
Size
7.66 GB

Product details

S1B_IW_SLC__1SDV_20200808T212338_20200808T212405_022837_02B594_E963.SAFE
Sensing date
9/8/2020
Cloud coverage
-
Availability
Immediate
Size
7.52 GB

Product details

S1B_IW_SLC__1SDV_20200813T213134_20200813T213202_022910_02B7C7_B7E3.SAFE
Sensing date
14/8/2020
Cloud coverage
-
Availability
Immediate
Size
7.66 GB

Product details

S1A_IW_SLC__1SDV_20200814T092343_20200814T092413_033901_03EE96_39D7.SAFE
Sensing date
14/8/2020
Cloud coverage
-
Availability
Immediate
Size
7.75 GB

Product details

S1A_IW_SLC__1SDV_20200709T092313_20200709T092343_033376_03DDF1_8F94.SAFE
Sensing date
9/7/2020
Cloud coverage
-
Availability
Immediate
Size
7.74 GB

Product details

S1A_IW_SLC__1SDV_20200721T092314_20200721T092344_033551_03E34E_66D2.SAFE
Sensing date
21/7/2020
Cloud coverage
-
Availability
Immediate
Size
7.74 GB

Product details

S1A_IW_SLC__1SDV_20200802T092315_20200802T092344_033726_03E8B0_676B.SAFE
Sensing date
2/8/2020
Cloud coverage
-
Availability
Immediate
Size
7.72 GB

Product details

S1A_IW_SLC__1SDV_20200814T092315_20200814T092345_033901_03EE96_F4C3.SAFE
Sensing date
14/8/2020
Cloud coverage
-
Availability
Immediate
Size
7.74 GB

Product details

S1A_IW_SLC__1SDV_20200702T093116_20200702T093146_033274_03DAEF_84AF.SAFE
Sensing date
2/7/2020
Cloud coverage
-
Availability
Immediate
Size
7.8 GB

Product details

S1B_IW_SLC__1SDV_20200720T213223_20200720T213250_022560_02AD16_6007.SAFE
Sensing date
21/7/2020
Cloud coverage
-
Availability
Immediate
Size
7.38 GB

Product details

S1B_IW_SLC__1SDV_20200801T213224_20200801T213251_022735_02B262_C9E6.SAFE
Sensing date
2/8/2020
Cloud coverage
-
Availability
Immediate
Size
7.38 GB

Product details

S1A_IW_SLC__1SDV_20200807T093118_20200807T093148_033799_03EB24_F1F6.SAFE
Sensing date
7/8/2020
Cloud coverage
-
Availability
Immediate
Size
7.8 GB

Product details

S1B_IW_SLC__1SDV_20200813T213224_20200813T213252_022910_02B7C7_567D.SAFE
Sensing date
14/8/2020
Cloud coverage
-
Availability
Immediate
Size
7.38 GB

Product details

S1A_IW_SLC__1SDV_20200702T093143_20200702T093211_033274_03DAEF_148B.SAFE
Sensing date
2/7/2020
Cloud coverage
-
Availability
Immediate
Size
7.25 GB

Product details

S1A_IW_SLC__1SDV_20200714T093144_20200714T093212_033449_03E046_B97F.SAFE
Sensing date
14/7/2020
Cloud coverage
-
Availability
Immediate
Size
7.25 GB

Product details

S1B_IW_SLC__1SDV_20200720T213158_20200720T213225_022560_02AD16_0047.SAFE
Sensing date
21/7/2020
Cloud coverage
-
Availability
Immediate
Size
7.38 GB

Product details

S1A_IW_SLC__1SDV_20200726T093145_20200726T093213_033624_03E5A6_591A.SAFE
Sensing date
26/7/2020
Cloud coverage
-
Availability
Immediate
Size
7.25 GB

Product details

S1B_IW_SLC__1SDV_20200801T213159_20200801T213226_022735_02B262_3E4E.SAFE
Sensing date
2/8/2020
Cloud coverage
-
Availability
Immediate
Size
7.38 GB

Product details

S1A_IW_SLC__1SDV_20200807T093146_20200807T093214_033799_03EB24_E562.SAFE
Sensing date
7/8/2020
Cloud coverage
-
Availability
Immediate
Size
7.25 GB

Product details

S1B_IW_SLC__1SDV_20200813T213200_20200813T213227_022910_02B7C7_EC64.SAFE

"""
safe_list = re.findall(r"S1[AB]_IW_SLC__1SDV_\d{8}T\d{6}_\d{8}T\d{6}_[\dA-F]{6}_\w{6}_\w{4}\.SAFE",
                       text_block, re.IGNORECASE)
safe_list = sorted(set(safe_list))
print(f"🔎 추출된 SAFE 개수: {len(safe_list)}")

# 2. 토큰 발급/갱신 함수
TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
             "protocol/openid-connect/token")
def get_token(payload):
    r = requests.post(TOKEN_URL, data=payload); r.raise_for_status(); return r.json()

tok = get_token({"client_id":"cdse-public","grant_type":"password",
                 "username":EMAIL,"password":PW})
ACCESS, REFRESH = tok["access_token"], tok["refresh_token"]
exp_at = time.time() + tok["expires_in"] - 60

def ensure_header():
    global ACCESS, REFRESH, exp_at
    if time.time() > exp_at:                                   # 만료 임박
        t = get_token({"client_id":"cdse-public","grant_type":"refresh_token",
                       "refresh_token":REFRESH})
        ACCESS, REFRESH = t["access_token"], t.get("refresh_token", REFRESH)
        exp_at = time.time() + t["expires_in"] - 60
    return {"Authorization": f"Bearer {ACCESS}"}

# 3. SAFE → UUID 매핑
CATALOG = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Name eq '{}'"
name2uuid = {}
for s in safe_list:
    r = requests.get(CATALOG.format(s))
    items = r.json()["value"]
    if not items:
        print(f"❌ UUID 없음: {s}")
    else:
        name2uuid[s] = items[0]["Id"]
print(f"✅ UUID 확보: {len(name2uuid)} / {len(safe_list)}")

# 4. 다운로드
ZIPPER = ("https://zipper.dataspace.copernicus.eu/odata/v1/"
          "Products({})/$value")
out_dir = pathlib.Path("S1_raw")
out_dir.mkdir(exist_ok=True)

for safe, uid in name2uuid.items():
    url     = ZIPPER.format(uid)
    out_zip = out_dir / f"{safe}.zip"
    tmp     = out_zip.with_suffix(".part")

    pos = tmp.stat().st_size if tmp.exists() else 0
    hdr = ensure_header()
    if pos: hdr["Range"] = f"bytes={pos}-"

    with requests.get(url, headers=hdr, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + pos
        mode  = "ab" if pos else "wb"

        with open(tmp, mode) as f, tqdm(total=total, initial=pos,
                                        unit="B", unit_scale=True,
                                        desc=safe[:25]) as bar:
            for chunk in r.iter_content(1024*1024):
                f.write(chunk)
                bar.update(len(chunk))
                hdr = ensure_header()                 # 루프 중 토큰 갱신

    for attempt in range(5):
        try:
            os.replace(tmp, out_zip)   # 이미 있으면 덮어쓰기
            break                      # 성공
        except OSError as e:
            print(f"파일 잠김? 1초 후 재시도 ({attempt+1}/5) →", e)
            time.sleep(1)
    else:
        print("⚠️ rename 5회 실패, .part 그대로 둡니다")

    print(f"✔ {safe} 다운로드 완료 → {out_zip}")


print("\n🎉 모든 SAFE 다운로드가 끝났습니다.")