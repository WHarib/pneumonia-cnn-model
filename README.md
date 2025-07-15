---
title: Pneumonia CNN Classifier API
emoji: 🫁
colorFrom: indigo   # ← stays
colorTo: green      # ← replace teal with an allowed colour
sdk: docker
app_file: app.py
pinned: false
---


# Pneumonia CNN Classifier (FastAPI)

This Space exposes a REST endpoint that classifies chest-X-ray images as **Pneumonia** or **Normal** using a TensorFlow CNN (~90 % accuracy).

**Endpoint**

```http
POST /predict   (multipart/form-data, field name = file)
