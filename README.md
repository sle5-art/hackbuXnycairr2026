# hackbuXnycairr2026
code
Testing testing testing...

# SensorGap

A live NYC PM2.5 map that shows how sure it is. Where sensors are too sparse, it says "not enough data" instead of guessing, and it ranks the 10 places where a new sensor would help most.


```bash
git clone https://github.com/sle5-art/hackbuXnycairr2026.git
cd hackbuXnycairr2026
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
#Opens the map
'''bash
python -m streamlit run app/dashboard.py
'''
#runs the tests 
'''bash
python -m pytest
'''
