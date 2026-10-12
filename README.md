# ✈️ الذكاء الاصطناعي للصيانة التنبؤية (Predictive Maintenance AI)
### التنبؤ بالعمر الإنتاجي المتبقي لمحركات الطائرات (Remaining Useful Life - RUL)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)

---

## 📌 1. نبذة عن المشروع (Project Overview)

يهدف هذا المشروع إلى بناء نظام **صيانة تنبؤية (Predictive Maintenance)** يعتمد على خوارزميات تعلم الآلة للتنبؤ بـ **العمر الإنتاجي المتبقي (Remaining Useful Life - RUL)** لمحركات الطائرات النفاثة التوربينية (Turbofan Engines).

تتيح الصيانة التنبؤية لمشغلي الطائرات والمهندسين:
1. **استباق الأعطال الكارثية** قبل وقوعها لضمان أعلى معايير السلامة الجوية.
2. **تقليل أوقات التوقف غير المخطط لها (Unscheduled Downtime)**.
3. **تحسين جداول الصيانة وتكاليف قطع الغيار** بناءً على الحالة الفعلية للمحرك (Condition-Based Maintenance).

---

## 📊 2. مجموعة البيانات (Dataset: NASA C-MAPSS FD004)

تم استخدام مجموعة بيانات المحاكاة المعيارية التابعة لوكالة ناسا **(NASA Commercial Modular Aero-Propulsion System Simulation - C-MAPSS)**، وتحديداً المجموعة الفرعية المعقدة **FD004**:

- **شروط التشغيل:** محاكاة 6 أوضاع تشغيلية مختلفة (Multi-Condition: Mach, Altitude, Throttle).
- **أنماط الفشل (Fault Modes):** نوعان من أنماط الأعطال (HPC Degradation + Fan Degradation).
- **أعمدة البيانات (26 متغيراً):**
  - `unit_number`: معرّف المحرك (Engine ID).
  - `time_in_cycles`: الدورة التشغيلية الحالية (Operational Cycle).
  - `setting_1`, `setting_2`, `setting_3`: متغيرات إعدادات التشغيل (Operational Settings).
  - `sensor_1` إلى `sensor_21`: قراءات حساسات درجات الحرارة، الضغط، سرعات الدوران، ونسب التدفق.

---

## ⚙️ 3. منهجية المعالجة وهندسة الميزات (Pipeline & Engineering)

1. **حساب العمر الإنتاجي المتبقي (RUL Calculation):**
   تم حساب قيمة الـ RUL لكل دورة تشغيلية بناءً على أقصى دورة وصل إليها المحرك قبل الفشل الكامل:
   $$\text{RUL} = \text{max\_cycles} - \text{time\_in\_cycles}$$

2. **تحديد سقف الـ RUL (Piecewise Linear Degradation at 125 cycles):**
   في المراحل التشغيلية الأولى، يكون المحرك سليماً تماماً ولا يُظهر تدهوراً قابلاً للقياس عبر الحساسات. لذلك، تم تطبيق مبدأ **Capping عند 125 دورة** (`RUL.clip(upper=125)`)، مما يحسن دقة النماذج ويمنع تضليلها بقيم افتراضية غير ملموسة.

3. **تقسيم البيانات لمنع تسرب البيانات (Engine-based Group Split):**
   استخدام `GroupShuffleSplit` عبر `unit_number` لضمان أن جميع دورات المحرك الواحد إما في مجموعة التدريب (Train) أو التحقق (Validation) أو الاختبار (Test)، وتجنب تسرب البيانات بين المراحل.

4. **تحليل مراحل حياة المحرك (Life-Stage Categorization):**
   - **المرحلة الحرجة (Critical):** $RUL \le 30$ (تتطلب تدخلاً فورياً للصيانة).
   - **مرحلة التحذير المتوسط (Middle):** $30 < RUL \le 75$ (جدولة صيانة وقائية).
   - **المرحلة الآمنة المبكرة (Early/Healthy):** $RUL > 75$.

---

## 🔬 4. النماذج المدربة والنتائج (Model Performance & Comparison)

تم تدريب ومقارنة عدة نماذج، مع تقييمها بمقاييس الخطأ القياسية:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

| النموذج (Model) | عدد الميزات (Features) | MAE (متوسط الخطأ المطلق) | RMSE (جذر متوسط مربع الخطأ) | الحالة |
| :--- | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Mean)** | 24 | 45.10 | 53.40 | نموذج مرجعي |
| **Random Forest Regressor** | **24** | **14.20** | **20.11** | **النموذج الأفضل (Best)** |
| **Gradient Boosting Regressor** | 24 | 15.89 | 21.60 | أداء ممتاز |
| **Random Forest (Top 5 Sensors)** | 5 | 15.50 | 21.78 | نموذج خفيف وسريع |
| **Random Forest (Top 3 Sensors)** | 3 | 16.69 | 23.56 | نموذج مصغر |

### تحليل الخطأ حسب المرحلة التشغيلية (Life-Stage Error Analysis):
يُظهر النموذج دقة أعلى بكثير في **المرحلة الحرجة (0-30 دورة)** مقارنة بالمراحل المبكرة، وهو السلوك الأنسب لعمليات الصيانة الواقعية لتقليل مخاطر الأعطال المفاجئة.

---

## 🔍 5. أهم الحساسات المؤثرة (Top Degradation Sensors)

أظهر تحليل أهمية الخصائص (`Feature Importance`) وتحليل التباين أن الحساسات الأكثر حساسية لتآكل وتدهور أجزاء المحرك هي:
1. `sensor_13` (Core Speed / Fan Speed Related)
2. `sensor_11` (Static Pressure / HPC Outlet Pressure)
3. `sensor_15` (Bypass Ratio)
4. `sensor_9` & `sensor_14` (Total Temperature / Enthalpy)

---

## 📈 6. التكامل مع Power BI (Power BI Dashboard Integration)

تم تصميم مخرجات النموذج لتتكامل مباشرة مع **Microsoft Power BI** من خلال استخراج ملف تنبؤات وحالة المحركات، لبناء لوحة تحكم تفاعلية تتضمن:
- **مؤشرات أداء أسطول المحركات (Fleet Overview KPIs):** عدد المحركات النشطة، المحركات الحرجة، متوسط RUL.
- **تنبيهات الصيانة الفورية (Urgent Maintenance Alerts):** تلوين شرطي (أحمر، أصفر، أخضر).
- **التشخيص التفصيلي للمحرك (Deep-dive Diagnostics):** مسار هبوط RUL وقراءات الحساسات لكل محرك عبر الزمن.
- **تقييم أداء نموذج الذكاء الاصطناعي (Model Accuracy vs Actuals):** مقارنة التنبؤ بالواقع.

*(انظر الدليل المفصل في الأسفل لطريقة إعداد لوحة Power BI خطوة بخطوة).*

---

## 💻 7. التثبيت والتشغيل (Setup & Execution)

### المتطلبات (Prerequisites):
```bash
pip install -r requirements.txt
```

### هيكل المشروع (Project Directory):
```text
Predictive Maintenance AI/
│
├── dashboard.py               # 🚀 واجهة المراقبة التفاعلية (Streamlit Dashboard)
├── untitled1.py               # ⚙️ كود المعالجة، التدريب، وتقييم النماذج (ML Pipeline)
├── export_for_powerbi.py      # 📊 تصدير البيانات بصيغة CSV لـ Power BI
├── requirements.txt           # 📦 مكتبات Python المطلوبة
├── .streamlit/config.toml     # 🎨 إعدادات ثيم الواجهة المظلمة
├── .github/workflows/ci.yml   # 🤖 GitHub Actions CI Pipeline
├── .gitignore                 # 🚫 ملفات مستثناة من Git
├── train_FD004.txt            # 📁 ملف بيانات ناسا C-MAPSS (غير مرفوع على Git)
└── README.md                  # 📖 توثيق المشروع
```

### 🚀 تشغيل واجهة المراقبة التفاعلية (Dashboard):
```bash
streamlit run dashboard.py
```
> يفتح المتصفح تلقائياً على `http://localhost:8501` وتظهر اللوحة بثيم داكن احترافي.
> الواجهة تعمل بـ **Demo Mode** تلقائياً إذا لم يتوفر ملف `train_FD004.txt`.

### تشغيل كود التدريب فقط:
```bash
python untitled1.py
```

### تصدير بيانات Power BI:
```bash
python export_for_powerbi.py
```

---

## 👥 فريق العمل والمساهمة
تم تطوير هذا المشروع كجزء من أنظمة الصيانة التنبؤية المعتمدة على الذكاء الاصطناعي.
إذا كان لديك أي استفسار أو اقتراح للتحسين، يسعدنا تواصلك!
