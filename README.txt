Source
https://www.youtube.com/watch?v=5yPeKQzCPdI&t=1190s 
https://pysource.com/2021/08/16/face-recognition-in-real-time-with-opencv-and-python/  [jeleknya ini pake internal library]

untuk bisa dilihat dari webcam follow this
https://www.youtube.com/watch?v=sz25xxF_AVE [ini dibuka semua caranya]

github
https://github.com/ageitgey/face_recognition

Package required

conda create -n face_recog python=3.9 
activate face_recogn
conda install pip
pip install opencv-python
pip install more-itertools numba tiktoken==0.3.3 torch tqdm openai-whisper

conda install python=3.9 downgrade your python # or do it during environment creation
pip install dlib-19.22.99-cp39-cp39-win_amd64.whl  (ada di dalam folder dlib)
pip install face_recognition
pip install deepface
pip install numpy --force-reinstall #reinstall numpy hanya jika kita downgrade python ditengah2 instalasi


additional_documents[experts]
to improve the accuraccy
https://github.com/ageitgey/face_recognition/wiki/Face-Recognition-Accuracy-Problems
https://pyimagesearch.com/2018/09/24/opencv-face-recognition/
