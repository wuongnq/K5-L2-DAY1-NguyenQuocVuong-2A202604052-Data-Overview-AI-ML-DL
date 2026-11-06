# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:** Python: 3.13.16, PyTorch: 2.11.0+cu130, Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

## 1. Phân loại ảnh

1. `class_id`=468 ; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. **Cả bức ảnh**: `task`=`image_classification`, không có `bbox_xyxy`/`polygon_xy`, chỉ có 5 lớp xếp hạng cho toàn ảnh.
3. **Đi kèm taxonomy/checkpoint** (ImageNet-1K, 1000 lớp). Model chỉ chọn trong danh sách này; vd. ảnh `kitchen` bị đoán `gong` vì taxonomy không có lớp "bếp".
4. Cùng tên lớp có thể khác ID giữa các taxonomy (vd. `dining_table` = 532 ở ImageNet-1K, khác ở COCO-80). Ghi đủ ba trường để tra ngược và tái lập được.
5. Ảnh `traffic` có top-5 toàn xe (cab, minibus, police_van…). Guideline cần nói: chọn **chủ thể chính** (lớn nhất/ở giữa), thứ tự ưu tiên khi ngang nhau, và làm gì khi không có lớp phù hợp.
6. Score chỉ là độ "tin" của model, có thể cao mà vẫn sai: `kitchen` → `gong` score 0.42. Ground truth do người gán theo guideline.

## 2. Phát hiện vật thể

1. `class_name`="person" ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Gốc (0,0) ở góc trên-trái ảnh, y tăng xuống dưới. Góc trên-trái hộp (385, 69), góc dưới-phải (499, 349); rộng ≈ 114 px, cao ≈ 280 px.
3. Ngưỡng 0.35: **11** prediction; ngưỡng 0.60: **6** prediction. 
4. Reviewer phải loại thêm hộp sai và hộp trùng (vd. hai bowl [0.32, 305.02, 87.69, 371.32] và [32.65, 342.12, 100.16, 384.93] chồng nhau), kiểm tra lại lớp.
5. *Mỗi vật một hộp, ôm sát phần nhìn thấy, không chừa nền; vật cắt mép thì kéo hộp sát mép ảnh.*
6. `person` [0.12, 263.18, 61.55, 310.78] chạm mép trái, chỉ thấy một phần người. → **Guideline** (quy tắc câu 5), vì cắt mép lặp lại ở nhiều ảnh. Chỉ escalation nếu không xác định được đó có phải người.

## 3. Phân đoạn theo từng đối tượng

1. `instance_id`="kitchen-001" ; `class_name`="person" ; `score`=0.899318 ; `polygon_point_count`=348 ; 5 điểm đầu=[[446,70], [445,71], [444,71], [443,72], [442,72]]
2. Polygon theo sát đường viền vật nên cho hình dạng thật. Với `kitchen-001`, polygon chỉ phủ ~57% diện tích box; phần còn lại là nền.
3. `instance_id` phân biệt từng vật trong **một ảnh**: 4 bowl cùng `class_id`=45 nhưng có ID riêng (`kitchen-002`, `-003`, `-006`, `-010`). Không phải class ID (đó là *loại* vật), không phải tracking ID (không theo vật qua nhiều ảnh/frame).
4. *Biên mask sát mép phần nhìn thấy; không tô phần bị che; mỗi pixel chỉ thuộc một instance (vật phía trước).*
5. Bàn ăn `kitchen-005` bị các bowl `kitchen-002/003/006` che, hai bowl `kitchen-002`/`-006` chồng nhau. → **Guideline** (quy tắc câu 4), vì vật đặt trên bàn rất phổ biến. Escalation nếu không tách được 1 hay 2 bát.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth (đơn vị nhãn) | Lỗi / chỗ khó | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 lớp cho cả ảnh | `kitchen` → `gong` 0.42 (sai) | Chọn 1 lớp theo quy tắc chủ thể chính | Lớp đúng taxonomy, đúng quy tắc ưu tiên |
| Phát hiện vật thể | 1 lớp + 1 box mỗi vật | `person` cắt mép trái; 2 bowl chồng hộp | Vẽ box ôm sát phần nhìn thấy | Box thừa/trùng/thiếu/lệch, sai lớp |
| Instance segmentation | 1 lớp + 1 polygon mỗi instance | Bàn `kitchen-005` bị bowl che | Tô mask sát biên, không chồng nhau | Biên lệch, mask chồng, gộp/tách sai instance |

## 5. An toàn dữ liệu

1. Chỉ dùng 3 ảnh COCO công khai của notebook; không đưa ảnh/dữ liệu cá nhân hay nội bộ vào báo cáo và repo.
2. Dừng lại và báo Lab Coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
