from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
import os
from app.inference import classify_fish
import numpy as np  # Import numpy để chuyển đổi ndarray

router = APIRouter()

@router.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Lưu tệp tin đã tải lên vào thư mục tạm thời
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        # Ghi tệp vào thư mục tạm thời
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Lớp phân loại cá từ hàm classify_fish
        label = classify_fish(file_path)

        # Nếu label là một mảng NumPy, chuyển nó thành list trước khi trả về
        if isinstance(label, np.ndarray):
            label = label.tolist()

        # Xóa tệp tin sau khi xử lý
        os.remove(file_path)

        # Trả về kết quả phân loại dưới dạng JSON response
        return JSONResponse(
            status_code=200,  # Mã trạng thái HTTP 200 cho yêu cầu thành công
            content={"label": label}  # Trả về dữ liệu phân loại dưới dạng JSON
        )

    except Exception as e:
        # Trả về thông báo lỗi nếu có ngoại lệ xảy ra
        print(f"Đã xảy ra lỗi: {str(e)}")
        return JSONResponse(
            status_code=500,  # Mã trạng thái HTTP 500 cho lỗi nội bộ
            content={"error": f"File processing failed: {str(e)}"}  # Trả về thông báo lỗi dưới dạng JSON
        )

# Khởi tạo FastAPI app và thêm router
app = FastAPI()

app.include_router(router)
