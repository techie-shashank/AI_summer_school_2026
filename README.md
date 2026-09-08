# AI_Summer_School_2026
Repository for Bavarian-Czech Summer School “AI and Industry” 2026

### Notes

#### Image to Base64 

```Python
image = cv2.imread(IMAGE_PATH)
_, buffer = cv2.imencode('.png', image)
img_base64 = base64.b64encode(buffer).decode('utf-8')
```

#### UV Python package manager instalation

https://docs.astral.sh/uv/getting-started/installation/

#### Competition laptop HW specs

- **CPU:** Intel Core i7-9750 2.6 GHz x 12
- **RAM:** 24 GB
- **GPU:** NVIDIA GeForce RTX 2060 Mobile. 6 GB VRAM 