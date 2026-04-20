A deep learning-based clinical decision support tool that classifies 7 types 
of skin lesions from dermoscopic images using a two-phase MobileNet model 
trained on the HAM10000 dataset (10,015 images).

Features:
- 7-class skin lesion classification (MEL, NV, BCC, AKIEC, BKL, DF, VASC)
- Gatekeeper model to reject non-skin images
- Multi-input inference: image + patient metadata (age, sex, location)
- Downloadable PDF diagnostic reports
- Patient history dashboard
- User authentication (register/login)
- Dark/light mode UI
- PostgreSQL (production) / SQLite (development)
- Docker support

Built with: Python, Flask, TensorFlow, MobileNet, PostgreSQL, TailwindCSS
Dataset: HAM10000 (ISIC Archive)
