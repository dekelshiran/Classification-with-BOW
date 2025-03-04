# Classification-with-BOW

# שאלה 1

חלקו את הנתונים לנתוני אימון – 
,train-set כאשר 60% מהתמונות ישמשו לאימון, 20% לוולידציה ו20%- לבדיקה.
שלב האימון – יתבצע על ה- train-set בלבד:

.1  חישוב מאפיינים מכל התמונות. )אפשר להשתמש בפונקציות לזיהוי נקודות ומאפיינים של
.(OpenCV
.2 Quantization Vector – חישוב K-means על סט המאפיינים. מספר המרכזים צריך להיות גדול מספיק )לפחות (100 ונתון לשיקולכם. שלב זה יכול להיות כבד חישובית. אפשר לדגום חלק מהווקטורים או דרך אחרת להקל על החישוב.
.3  לכל תמונה חישוב היסטוגרמה של מספר המאפיינים השייכים לcluster- ה.k-
.4 אימון מסווג לבחירתכם כאשר כל תמונה מיוצגת ע"י ההיסטוגרמה שחושבה ב(3)- ולכל תמונה מוצמד label שמשייך אותה למחלקה המתאימה.

כתוצאה משלב האימון, המודל מורכב מהמרכזים שחושבו בשלב הVQ- ע"י K-means והמסווג שאומן בשלב .(4)
שלב הבדיקה – יתבצע על ה- test-set בלבד לכל תמונה -
.1  חישוב מאפיינים בצורה זהה לשלב האימון.
.2 Quantization Vector – לכל מאפיין חישוב המרכז הקרוב ביותר. .3  חישוב היסטוגרמה של מספר המאפיינים בכל מרכז.
 
.4  סיווג ע"י המסווג שאומן בשלב האימון.

מה הפרמטרים האופטימליים של המסווג? יש להציג את הביצועים ע"י עקומות ,ROC חישוב ה,AUC-
.confusion matrix-ו ,precision-recall עקומות

 




 # שאלה 2
# BoW with Deep Learning Features 

בחלק זה נממש ייצוג BOW ע"י חישוב מפות מאפיינים מ.CNN-
כדי לחשב מאפיינים ע"י רשת VGG שאומנה על ImageNet נשתמש בקוד הבא:
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms from PIL import Image

Load a pre-trained vgg-16 model model = models.vgg16(pretrained=True)

Remove the classifier (fully connected layers) to get the feature map from the last conv layer feature_extractor = nn.Sequential(*list(model.children())[:-2])

Set the feature extractor to evaluation mode feature_extractor.eval()

Image transformation (maintain original size) transform = transforms.Compose([
transforms.ToTensor(), # Convert to tensor transforms.Normalize( # Normalize to ImageNet mean and std mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)
])

Load an image (original size is maintained)
image = Image.open("example.jpg").convert("RGB") # Replace with your image path input_tensor = transform(image).unsqueeze(0) # Add batch dimension

Forward pass through the feature extractor feat_map = feature_extractor(input_tensor)
המוצא feat_map הוא טנסור 4 ממדי - ,8[ ,8 ,512 ]1 . שמייצג 8X8 ווקטורים בגודל 512 שמפוזרים אחיד ע"פ התמונה. הווקטורים האלו ישמשו ליצירת ייצוג BOW בצורה זהה לשימוש של המאפיינים בשאלה הקודמת.
בעזרת המאפיינים האלו מסט האימון יש לבנות את המילון ולאמן מסווג. יש להראות את התוצאות בפורמט דומה לשאלה הקודמת.
