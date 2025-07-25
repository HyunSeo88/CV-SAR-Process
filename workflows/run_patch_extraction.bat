@echo off
echo ============================================================
echo Sentinel-1 패치 추출기 실행 스크립트
echo ============================================================
echo.

:: 환경 설정
set "CONDA_ENV=snap"
set "JAVA_OPTS=-Xms2048m -Xmx16384m"
set "SNAP_OPTS=-Xms2048m -Xmx16384m"

echo 1. 환경 설정 중...
call conda activate %CONDA_ENV%
if %errorlevel% neq 0 (
    echo ❌ Conda 환경 활성화 실패: %CONDA_ENV%
    echo    conda env list 명령으로 사용 가능한 환경을 확인하세요.
    pause
    exit /b 1
)

echo ✅ Conda 환경 활성화 완료: %CONDA_ENV%
echo ✅ Java 메모리 설정: %JAVA_OPTS%
echo.

:: 워킹 디렉토리 변경
cd /d "%~dp0"
echo 2. 작업 디렉토리: %CD%
echo.

:: 사용자 선택
echo 실행할 스크립트를 선택하세요:
echo [1] 디버깅 테스트 (10개 패치만 생성)
echo [2] 전체 패치 추출
echo [3] 종료
echo.
set /p choice="선택 (1-3): "

if "%choice%"=="1" goto debug_test
if "%choice%"=="2" goto full_extraction
if "%choice%"=="3" goto end
echo 잘못된 선택입니다.
goto end

:debug_test
echo.
echo ============================================================
echo 디버깅 테스트 시작
echo ============================================================
python patch_extractor_debug.py
if %errorlevel% equ 0 (
    echo.
    echo ✅ 디버깅 테스트 완료!
    echo 결과 확인: D:\Sentinel-1\data\processed_2_test\
) else (
    echo.
    echo ❌ 디버깅 테스트 실패
)
goto end

:full_extraction
echo.
echo ============================================================
echo 전체 패치 추출 시작
echo ============================================================
echo ⚠️ 주의: 이 작업은 매우 오래 걸릴 수 있습니다 (몇 시간~하루)
echo.
set /p confirm="계속하시겠습니까? (y/N): "
if /i not "%confirm%"=="y" goto end

echo.
echo 패치 추출을 시작합니다...
echo 로그 파일: patch_extraction.log
echo.

python patch_extractor.py
if %errorlevel% equ 0 (
    echo.
    echo ✅ 패치 추출 완료!
    echo 결과 확인: D:\Sentinel-1\data\processed_2\
    echo 로그 확인: patch_extraction.log
) else (
    echo.
    echo ❌ 패치 추출 실패
    echo 로그 파일을 확인하세요: patch_extraction.log
)
goto end

:end
echo.
echo 스크립트 실행 완료.
pause 