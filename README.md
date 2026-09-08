# AI_Summer_School_2026
Repository for Bavarian-Czech Summer School “AI and Industry” 2026


### Notes:

#### Image to Base64 

```Python
image = cv2.imread(IMAGE_PATH)
_, buffer = cv2.imencode('.png', image)
img_base64 = base64.b64encode(buffer).decode('utf-8')
```

### UV Python package manager instalation

https://docs.astral.sh/uv/getting-started/installation/
