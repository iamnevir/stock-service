# Hướng dẫn Setup Server với Docker

Dưới đây là các bước quy trình setup server tự động đã được Dockerize.

## Yêu cầu trước khi cài đặt
- Đã cài đặt **Docker** và **Docker Compose**.
- Đã cài đặt **Git**.

## Các bước triển khai

### Bước 1: Clone thư mục từ GitHub
Mở terminal tại server và chạy lệnh sau để clone project:
```bash
git clone <url_repo_cua_ban> nevir
cd nevir
```

### Bước 2 & 3 & 4 & 5 & 6: Khởi chạy bằng Docker
Toàn bộ quá trình cài đặt MongoDB, bật Auth, thiết lập PYTHONPATH, chạy file `auto/test.py` để tạo index MongoDB, và khởi chạy API server (`api/api.py`) đã được tự động hoá trong `docker-compose.yml`.

Chỉ cần chạy lệnh sau tại thư mục chứa file `docker-compose.yml`:
```bash
docker-compose up -d --build
```
*(Nếu hệ thống dùng phiên bản Docker mới, bạn có thể dùng lệnh `docker compose up -d --build`)*

**Lưu ý về SSH Key:** 
Container được cấu hình mount thư mục `~/.ssh` của host vào `/root/.ssh` ở chế độ read-only, do đó container sẽ có thể sử dụng các cấu hình SSH keys (để kéo code hoặc kết nối với server khác) như host của bạn. Hãy đảm bảo bạn đã tạo và cấu hình SSH keys trên máy host.

---

## ⚠️ Lưu ý CỰC KỲ QUAN TRỌNG
Bạn **BẮT BUỘC** phải báo với admin hệ thống / người quản lý server chính để **thêm (allow) địa chỉ IP của server này vào whitelist (tường lửa)** của server chính. Nếu không, server này sẽ không thể giao tiếp được với hệ thống MongoDB ở server trung tâm.

Ví dụ lệnh thêm IP (thực hiện ở server chính):
```bash
sudo ufw allow in proto tcp from <IP_SERVER_MỚI_NÀY> to any port 27017 
```
