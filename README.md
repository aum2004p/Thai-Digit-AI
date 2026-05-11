# Thai-Digit-AI (51-55)

Members
1) ณัฐริกา บุญส่ง 1660701846
2) มินทร์ตรา พรรณเขียวหวาน 1660702166
3) ธิดารัตน์ วิเชียร 1660702927
4) วรินทร์ ชาติชนบท 1660704865
5) พิมพ์ตะวัน ทองจูด 1660706233

วัตถุประสงค์
พัฒนาระบบรู้จำเลขไทยด้วย AI
สร้าง dataset สำหรับฝึกโมเดล
พัฒนาเว็บแอปสำหรับใช้งานจริง
แสดงผลการทำนายแบบ real-time


📊 System Architecture (Table Format)
| Step | Component               | Role               | Input                   | Process                                           | Output                               |
| ---- | ----------------------- | ------------------ | ----------------------- | ------------------------------------------------- | ------------------------------------ |
| 1    | Web Browser (User)      | User Interface     | Mouse drawing on canvas | Draw Thai digit (51–55)                           | Base64 encoded image                 |
| 2    | Flask Web Server        | Backend API Server | Base64 image            | Receive request at `/predict` endpoint            | Raw image data                       |
| 3    | Image Preprocessing     | Data Preparation   | Raw image               | Resize (32×32), grayscale, normalize              | Tensor format                        |
| 4    | PyTorch CNN Model (.pt) | AI Model           | Processed tensor        | Convolutional feature extraction + classification | Predicted class (51–55)              |
| 5    | Prediction Engine       | Result Handler     | Model output            | Softmax probability calculation                   | Label + confidence scores            |
| 6    | JSON Response           | API Response Layer | Prediction result       | Format response as JSON                           | `{label, confidence, probabilities}` |
| 7    | Frontend Display        | UI Result Renderer | JSON response           | Show result + probability bars                    | Prediction UI output                 |



Features
UserPage : ผู้ใช้สามารถวาดตัวเลขไทย 51-55 เพื่อให้ AI ทำนายผลว่าคือเลขอะไร
AdminPage : - สามารถอัปโหลดตัวโมเดลได้ใหม่โดยใช้ไฟล์โมเดลนามสกุล .pt
            - ดูรายการโมเดลที่อัปโหลดได้ฃ
            - ทดสอบทำนายจากรูปภาพ
            - สามารถอัปโฆลดรูปภาพหรือเอาจากรูปที่ผู้ใช้ได้ทำการเขียนไว้เพื่อ Train/Test ได้
            - จะเก็บรูปภาพที่ผู้ใช้ได้ทำการเขียนทั้งหมด

            
เทคโนโลยีที่ใช้
🔬 Machine Learning
PyTorch (Deep Learning Framework)
CNN (Convolutional Neural Network)

🖼️ Image Processing
Pillow (PIL)
NumPy

🌐 Web Development
Flask (Backend API)
HTML / CSS / JavaScript (Frontend)

🧪 Tools
Python 3.14
VS Code
Git / GitHub

วิธีการใช้งาน
รันระบบ
cd thai-digit-recognition/webapp
python app.py

จะได้ http://192.168.1.105:5000

Libraries ที่ใช้
torch
torchvision
flask
pillow
numpy
werkzeug

สรุป
เป็นระบบ AI Web Application ที่สามารถรับ input จากผู้ใช้แบบ real-time และทำการทำนายเลขไทยด้วยโมเดล Deep Learning โดยใช้ PyTorch ร่วมกับ Flask ซึ่งสามารถนำไปต่อยอดในระบบ OCR หรือ AI education ได้

จุดเด่นของระบบ
ใช้งานผ่านเว็บได้ทันที
real-time prediction
ใช้ AI model จริง (PyTorch)
มีระบบ admin อัปโหลดโมเดล
