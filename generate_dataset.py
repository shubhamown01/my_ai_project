"""
global_object_dataset.py  —  10,000 unique synthetic product records
=====================================================================
References:
  • Google Product Taxonomy 2021
  • Amazon BTG attribute logic
  • GS1 GPC Brick Codes + material/feature standards
  • Schema.org Product Vocabulary (brand, model, mpn, gtin, material, color)
  • COCO 80-class YOLOv8 mapping
  • Material Measurement Standards (texture/finish properties)
"""

import os, random, hashlib
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────
# 1. YOLO / COCO 80 classes
# ──────────────────────────────────────────────────────────────
YOLO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]

# ──────────────────────────────────────────────────────────────
# 2. 100+ Real-World Brands  (Schema.org: brand)
# ──────────────────────────────────────────────────────────────
BRANDS = {
    "Electronics": [
        "Apple","Samsung","Sony","LG","Xiaomi","OnePlus","Asus","Dell","HP",
        "Lenovo","Acer","MSI","Razer","Logitech","JBL","Bose","Sennheiser",
        "Philips","Panasonic","Toshiba","Huawei","Oppo","Vivo","Realme",
        "Nothing","Google","Microsoft","Intel","AMD","Nvidia",
    ],
    "Apparel & Accessories": [
        "Nike","Adidas","Puma","Reebok","Under Armour","Levi's","H&M","Zara",
        "Uniqlo","Gap","Calvin Klein","Tommy Hilfiger","Gucci","Prada","Versace",
        "Burberry","Lacoste","New Balance","Converse","Vans","Fila","Asics",
        "Columbia","The North Face","Patagonia",
    ],
    "Kitchen & Home": [
        "Milton","Stanley","Thermos","Prestige","Borosil","Bajaj","Pigeon",
        "Tefal","Cuisinart","KitchenAid","Instant Pot","Vitamix","Ninja",
        "OXO","Joseph Joseph","Le Creuset","Lodge","Calphalon","Ikea",
        "Philips","Morphy Richards","Hamilton Beach","Black+Decker","Braun",
    ],
    "Hardware & Tools": [
        "Bosch","Makita","DeWalt","Milwaukee","Stanley","Black+Decker","Ryobi",
        "Hilti","Festool","Metabo","Snap-on","Klein Tools","Irwin","Knipex",
        "Wera","Facom","Bahco","Leatherman","Gerber","Buck",
    ],
    "Sporting Goods": [
        "Nike","Adidas","Puma","Wilson","Head","Yonex","Callaway","TaylorMade",
        "Titleist","Ping","Rawlings","Easton","Louisville Slugger","Speedo",
        "Arena","Brooks","Salomon","Rossignol","Atomic","Fischer",
    ],
    "Food & Beverage": [
        "Nestle","Coca-Cola","Pepsi","Lay's","Pringles","Kellogg's","Heinz",
        "Maggi","Amul","Britannia","Haldiram's","Parle","ITC","Cadbury",
        "Mars","Ferrero","Hershey's","Kraft","Unilever","P&G",
    ],
    "Furniture": [
        "Ikea","Ashley","La-Z-Boy","Ethan Allen","Crate & Barrel","West Elm",
        "Pottery Barn","Herman Miller","Steelcase","Knoll","Haworth",
        "Urban Ladder","Pepperfry","Godrej","Nilkamal","Durian",
    ],
    "Toys & Games": [
        "Lego","Hasbro","Mattel","Fisher-Price","Hot Wheels","Barbie",
        "Nerf","Play-Doh","Funko","Ravensburger","Melissa & Doug",
        "VTech","LeapFrog","Bandai","Namco",
    ],
    "Baby & Toddler": [
        "Johnson & Johnson","Pampers","Huggies","Chicco","Graco","Mamas & Papas",
        "Fisher-Price","Infantino","Skip Hop","Maxi-Cosi","Joie","Nuk","Avent",
    ],
    "Vehicles": [
        "Toyota","Honda","Ford","BMW","Mercedes-Benz","Audi","Volkswagen",
        "Hyundai","Kia","Nissan","Suzuki","Tata","Mahindra","Hero","Bajaj",
    ],
}
ALL_BRANDS = [b for lst in BRANDS.values() for b in lst]

# ──────────────────────────────────────────────────────────────
# 3. Colors  (Schema.org: color)
# ──────────────────────────────────────────────────────────────
COLORS = [
    "Midnight Black","Polar White","Space Gray","Rose Gold","Midnight Blue",
    "Forest Green","Coral Red","Titanium Silver","Champagne Gold","Slate Gray",
    "Navy Blue","Burgundy","Olive Green","Powder Pink","Electric Blue",
    "Matte Black","Pearl White","Graphite","Sand Beige","Cobalt Blue",
    "Emerald Green","Crimson Red","Obsidian","Ivory","Sapphire Blue",
    "Onyx Black","Arctic White","Bronze","Teal","Lavender",
    "Turquoise","Charcoal","Orange","Purple","Yellow","Lime Green",
    "Magenta","Cyan","Maroon","Khaki",
]

# ──────────────────────────────────────────────────────────────
# 4. GS1 GPC Materials + Material Measurement Standards
# ──────────────────────────────────────────────────────────────
MATERIALS = {
    "Electronics":  ["Brushed Aluminum","Polycarbonate","Tempered Glass",
                     "Stainless Steel","Titanium Alloy","Carbon Fiber",
                     "Anodized Aluminum","Gorilla Glass","Ceramic"],
    "Apparel":      ["100% Cotton","Polyester Blend","Merino Wool",
                     "Nylon Ripstop","Vegan Leather","Genuine Leather",
                     "Linen","Bamboo Fiber","Modal","Spandex Blend",
                     "Recycled PET","Fleece","Denim"],
    "Kitchen":      ["Food-Grade Stainless Steel","Borosilicate Glass",
                     "BPA-Free Polypropylene","Cast Iron","Ceramic Coated",
                     "Hard-Anodized Aluminum","Copper Clad","Melamine",
                     "Silicone","Bamboo"],
    "Tools":        ["Hardened Steel","Chrome Vanadium Steel","Alloy Steel",
                     "Forged Aluminum","Fiberglass Reinforced","ABS Plastic",
                     "Rubber Grip","Titanium Coated","Bi-Material"],
    "Furniture":    ["Solid Oak Wood","MDF with Veneer","Tempered Glass",
                     "Powder-Coated Steel","Solid Walnut","Plywood",
                     "Acacia Wood","Bamboo","Rattan","Foam + Fabric"],
    "Sporting":     ["Polycarbonate Shell","EVA Foam","Natural Rubber",
                     "Neoprene","Graphite Composite","Carbon Fiber Shaft",
                     "Thermoplastic Urethane","Mesh Fabric","Leather"],
    "Default":      ["Plastic","Rubber","Metal","Wood","Glass","Fabric",
                     "Silicone","Paper","Ceramic","Composite"],
}

# ──────────────────────────────────────────────────────────────
# 5. Spec-generation functions (BTG attribute logic)
# ──────────────────────────────────────────────────────────────
def spec_electronics(brand, model, variant):
    ram     = random.choice(["4GB","6GB","8GB","12GB","16GB","32GB","64GB"])
    storage = random.choice(["64GB","128GB","256GB","512GB","1TB","2TB"])
    proc    = random.choice(["Snapdragon 8 Gen3","Apple A17 Pro","Dimensity 9300",
                              "Intel Core i9","AMD Ryzen 9","Intel Core i7",
                              "MediaTek Helio G99","Kirin 9000","Exynos 2400"])
    battery = random.choice(["3000mAh","4500mAh","5000mAh","6000mAh"])
    res     = random.choice(["Full HD 1080p","2K QHD","4K UHD","8K","WQHD+"])
    conn    = random.choice(["Wi-Fi 6E","Wi-Fi 7","Bluetooth 5.3","5G + Wi-Fi 6"])
    refresh = random.choice(["60Hz","90Hz","120Hz","144Hz","165Hz","240Hz"])
    return (f"Processor:{proc} | RAM:{ram} | Storage:{storage} | "
            f"Battery:{battery} | Display:{res} | Refresh:{refresh} | Connectivity:{conn}")

def spec_apparel(brand, model, variant):
    size    = random.choice(["XS","S","M","L","XL","XXL","3XL"])
    fit     = random.choice(["Slim Fit","Regular Fit","Relaxed Fit","Athletic Fit","Oversized"])
    care    = random.choice(["Machine Wash Cold","Hand Wash Only","Dry Clean Only"])
    origin  = random.choice(["Made in India","Made in Bangladesh","Made in Vietnam",
                              "Made in China","Made in Turkey","Made in Italy"])
    return f"Size:{size} | Fit:{fit} | Care:{care} | {origin}"

def spec_kitchen(brand, model, variant):
    cap     = random.choice(["250ml","500ml","750ml","1L","1.5L","2L","3L","5L"])
    weight  = random.choice(["0.3kg","0.5kg","0.8kg","1.2kg","2.5kg","4kg"])
    temp    = random.choice(["Up to 120C","Up to 180C","Up to 250C","Freezer Safe"])
    power   = random.choice(["500W","750W","1000W","1200W","1500W","2000W","2200W"])
    warr    = random.choice(["1 Year","2 Years","5 Years","Lifetime"])
    return (f"Capacity:{cap} | Weight:{weight} | Max Temp:{temp} | "
            f"Power:{power} | Warranty:{warr}")

def spec_tools(brand, model, variant):
    volt    = random.choice(["12V","18V","20V","36V","54V","Corded"])
    torque  = random.choice(["30Nm","60Nm","100Nm","200Nm","350Nm","500Nm"])
    speed   = random.choice(["0-500RPM","0-1500RPM","0-2000RPM","0-3000RPM"])
    chuck   = random.choice(["10mm Keyless","13mm Keyless","1/2 SDS","3/8 Keyless"])
    ip      = random.choice(["IP54","IP55","IP65","No Rating"])
    return f"Voltage:{volt} | Torque:{torque} | Speed:{speed} | Chuck:{chuck} | Protection:{ip}"

def spec_furniture(brand, model, variant):
    d = (f"{random.randint(40,200)}cm x "
         f"{random.randint(30,100)}cm x "
         f"{random.randint(30,200)}cm")
    weight  = f"{random.randint(3,80)}kg"
    load    = f"Max {random.randint(50,500)}kg"
    ass     = random.choice(["Self Assembly","Pre-Assembled","Professional Install"])
    return f"Dimensions:{d} | Weight:{weight} | Load:{load} | Assembly:{ass}"

def spec_sporting(brand, model, variant):
    weight  = random.choice(["150g","200g","250g","300g","400g","500g","1kg","2kg"])
    size    = random.choice(["Size 3","Size 4","Size 5","Small","Medium","Large","XL",
                              "24in","26in","27.5in","29in"])
    cert    = random.choice(["FIFA Approved","ITF Certified","ICC Approved",
                              "USGA Conforming","CE Certified","ISI Marked"])
    return f"Weight:{weight} | Size:{size} | Certification:{cert}"

def spec_food(brand, model, variant):
    weight  = random.choice(["50g","100g","200g","250g","500g","1kg","2kg"])
    cal     = random.choice(["80kcal","120kcal","200kcal","350kcal","500kcal"])
    shelf   = random.choice(["3 months","6 months","12 months","18 months","24 months"])
    store   = random.choice(["Store in Cool Dry Place","Refrigerate After Opening",
                              "Keep Frozen","Store Below 25C"])
    return f"Net Weight:{weight} | Calories:{cal} | Shelf Life:{shelf} | Storage:{store}"

def spec_toy(brand, model, variant):
    age     = random.choice(["3+","5+","8+","10+","12+","14+","18+"])
    pieces  = random.choice(["1 piece","42 pieces","100 pieces","250 pieces",
                              "500 pieces","1000 pieces"])
    batt    = random.choice(["3xAA Batteries","2xAAA Batteries","USB Rechargeable",
                              "No Battery Required","Solar Powered"])
    return f"Age:{age} | Pieces:{pieces} | Power:{batt}"

def spec_vehicle(brand, model, variant):
    engine  = random.choice(["1.0L Petrol","1.2L Petrol","1.5L Petrol","2.0L Diesel",
                              "Electric 40kWh","Electric 75kWh","Hybrid 1.8L"])
    power   = random.choice(["65HP","82HP","100HP","120HP","150HP","200HP","300HP"])
    mile    = random.choice(["15kmpl","18kmpl","22kmpl","25kmpl","30kmpl","400km range"])
    cols    = f"{random.randint(5,12)} color options"
    return f"Engine:{engine} | Power:{power} | Mileage:{mile} | {cols}"

def spec_baby(brand, model, variant):
    age     = random.choice(["0-6 months","0-12 months","1-3 years","3-5 years"])
    mat     = random.choice(["BPA-Free","Hypoallergenic","Organic Cotton","Food-Grade Silicone"])
    cert    = random.choice(["BIS Certified","CE Marked","FDA Approved","EN71 Compliant"])
    return f"Age:{age} | Safety:{mat} | Certification:{cert}"

def spec_default(brand, model, variant):
    return f"Model:{model} | Variant:{variant} | Brand:{brand}"

# ──────────────────────────────────────────────────────────────
# 6. Full Category Catalog
# ──────────────────────────────────────────────────────────────
CATEGORIES = [
    # ── Electronics ──────────────────────────────────────────
    {"gpc_brick":"10000233","gpc_name":"Smartphones",
     "google_path":"Electronics > Communications > Telephony > Mobile Phones",
     "short":"Electronics","yolo":["cell phone"],"mat":"Electronics","brand":"Electronics",
     "products":["iPhone","Galaxy S","Pixel","Redmi Note","OnePlus","Realme GT","Moto G",
                 "Poco X","Vivo V","Oppo Reno","Nothing Phone","Asus Zenfone",
                 "Huawei P","Xiaomi 14","Nokia G"],
     "models":["Pro","Ultra","Plus","Max","Mini","Lite","SE","FE","Neo","Turbo"],
     "spec_fn":spec_electronics,
     "desc_tmpl":"The {brand} {name} {variant} is a flagship {cat} device featuring cutting-edge performance, stunning {res} display, and advanced camera system."},

    {"gpc_brick":"10000234","gpc_name":"Laptops",
     "google_path":"Electronics > Computers > Laptops",
     "short":"Electronics","yolo":["laptop"],"mat":"Electronics","brand":"Electronics",
     "products":["MacBook Pro","MacBook Air","XPS","ThinkPad","Spectre","Envy",
                 "Surface Laptop","Gram","ZenBook","VivoBook","Aspire","Nitro",
                 "Blade","IdeaPad","Pavilion"],
     "models":["14in","15.6in","16in","13in","13.3in","17in","2-in-1","Touch"],
     "spec_fn":spec_electronics,
     "desc_tmpl":"The {brand} {name} {variant} laptop delivers exceptional performance with a sleek design, powerful internals, and all-day battery life."},

    {"gpc_brick":"10000235","gpc_name":"Tablets",
     "google_path":"Electronics > Computers > Tablet Computers",
     "short":"Electronics","yolo":["laptop"],"mat":"Electronics","brand":"Electronics",
     "products":["iPad Pro","iPad Air","Galaxy Tab","Surface Pro","MediaPad",
                 "Mi Pad","Lenovo Tab","Fire HD","MatePad","Redmi Pad"],
     "models":["WiFi","LTE","WiFi + Cellular","Kids Edition","Pro","Lite"],
     "spec_fn":spec_electronics,
     "desc_tmpl":"The {brand} {name} {variant} tablet combines portability with powerful performance for work, creativity, and entertainment."},

    {"gpc_brick":"10000240","gpc_name":"Televisions",
     "google_path":"Electronics > Video > Televisions",
     "short":"Electronics","yolo":["tv"],"mat":"Electronics","brand":"Electronics",
     "products":["OLED TV","QLED TV","NanoCell TV","Crystal UHD","Bravia XR",
                 "ULED TV","MiniLED TV","The Frame","Lifestyle TV","Neo QLED"],
     "models":["43in","50in","55in","65in","75in","85in","98in"],
     "spec_fn":spec_electronics,
     "desc_tmpl":"The {brand} {name} {variant} television delivers breathtaking visuals with vibrant colors, deep blacks, and immersive sound experience."},

    {"gpc_brick":"10000241","gpc_name":"Headphones",
     "google_path":"Electronics > Audio > Audio Components > Headphones & Headsets",
     "short":"Electronics","yolo":["person"],"mat":"Electronics","brand":"Electronics",
     "products":["WH-1000XM","QuietComfort","Momentum","HD 660","ATH-M50x",
                 "AirPods Pro","Freebuds Pro","Galaxy Buds","Redmi Buds","Jabra Elite"],
     "models":["Over-Ear","In-Ear","True Wireless","Neckband","Sport","Studio","ANC"],
     "spec_fn":spec_electronics,
     "desc_tmpl":"The {brand} {name} {variant} headphones offer superior audio quality with active noise cancellation and premium comfort for all-day wear."},

    {"gpc_brick":"10000242","gpc_name":"Smart Watches",
     "google_path":"Apparel & Accessories > Jewelry > Watches",
     "short":"Electronics","yolo":["cell phone"],"mat":"Electronics","brand":"Electronics",
     "products":["Apple Watch","Galaxy Watch","Pixel Watch","Mi Watch","Amazfit GTR",
                 "Fitbit Sense","Garmin Fenix","TicWatch","Huawei Watch","realme Watch"],
     "models":["Series 9","Ultra 2","Pro","Active","Classic","Sport","SE","R"],
     "spec_fn":spec_electronics,
     "desc_tmpl":"The {brand} {name} {variant} smartwatch seamlessly blends style with health tracking, fitness monitoring, and smart connectivity features."},

    {"gpc_brick":"10000243","gpc_name":"Desktop Computers",
     "google_path":"Electronics > Computers > Desktop Computers",
     "short":"Electronics","yolo":["tv"],"mat":"Electronics","brand":"Electronics",
     "products":["iMac","Mac Mini","Mac Pro","Inspiron Desktop","Pavilion Desktop",
                 "Legion Tower","Predator Orion","ROG Strix","OptiPlex","AiO Desktop"],
     "models":["Gaming","Creator","Business","Home","Workstation","Mini PC","Tower"],
     "spec_fn":spec_electronics,
     "desc_tmpl":"The {brand} {name} {variant} desktop delivers workstation-class performance in a compact or full-tower form factor."},

    {"gpc_brick":"10000244","gpc_name":"Keyboards",
     "google_path":"Electronics > Electronics Accessories > Computer Components > Input Devices > Keyboards",
     "short":"Electronics","yolo":["keyboard"],"mat":"Electronics","brand":"Electronics",
     "products":["MX Keys","G915","BlackWidow","Huntsman","K70","Anne Pro",
                 "Ducky One","GMMK Pro","Royal Kludge","Filco Majestouch"],
     "models":["TKL","Full Size","75 Percent","65 Percent","60 Percent","Wireless","RGB","Mechanical","Membrane"],
     "spec_fn":spec_electronics,
     "desc_tmpl":"The {brand} {name} {variant} keyboard provides a satisfying typing experience with programmable keys and responsive switches."},

    {"gpc_brick":"10000245","gpc_name":"Computer Mouse",
     "google_path":"Electronics > Electronics Accessories > Computer Components > Input Devices > Mice & Trackballs",
     "short":"Electronics","yolo":["mouse"],"mat":"Electronics","brand":"Electronics",
     "products":["MX Master","G502","DeathAdder","Viper","Basilisk","Model O",
                 "GPX Superlight","Aerox","Rival","Kone XP"],
     "models":["Wired","Wireless","Bluetooth","Gaming","Ergonomic","Vertical","Silent"],
     "spec_fn":spec_electronics,
     "desc_tmpl":"The {brand} {name} {variant} mouse delivers precision tracking with customizable buttons and ergonomic design for extended sessions."},

    {"gpc_brick":"10000246","gpc_name":"Remote Controls",
     "google_path":"Electronics > Electronics Accessories > Remote Controls",
     "short":"Electronics","yolo":["remote"],"mat":"Electronics","brand":"Electronics",
     "products":["Fire TV Remote","Harmony Elite","SofaBaton","Inteset","GE Universal",
                 "Logitech Harmony","One For All","Broadlink","BN59 Remote","AKB Remote"],
     "models":["Universal","Smart","Voice","Backlit","RF","IR","Bluetooth","4-Device"],
     "spec_fn":spec_electronics,
     "desc_tmpl":"The {brand} {name} {variant} remote control offers seamless device management with programmable buttons and ergonomic grip."},

    {"gpc_brick":"10000247","gpc_name":"Microwave Ovens",
     "google_path":"Home & Garden > Kitchen & Dining > Kitchen Appliances > Microwave Ovens",
     "short":"Electronics","yolo":["microwave"],"mat":"Electronics","brand":"Electronics",
     "products":["Convection Microwave","Solo Microwave","Grill Microwave",
                 "OTG Microwave","Smart Microwave","Countertop Microwave"],
     "models":["17L","20L","23L","25L","28L","30L","32L","Solo","Grill","Convection"],
     "spec_fn":spec_kitchen,
     "desc_tmpl":"The {brand} {name} {variant} microwave oven delivers fast, even cooking with multiple preset modes and child lock safety features."},

    {"gpc_brick":"10000248","gpc_name":"Refrigerators",
     "google_path":"Home & Garden > Kitchen & Dining > Kitchen Appliances > Refrigerators",
     "short":"Electronics","yolo":["refrigerator"],"mat":"Electronics","brand":"Electronics",
     "products":["Double Door Fridge","Side-by-Side Fridge","Single Door Fridge",
                 "French Door Fridge","Mini Fridge","Bottom Freezer Fridge"],
     "models":["195L","253L","308L","415L","450L","516L","655L","Frost Free","Direct Cool"],
     "spec_fn":spec_kitchen,
     "desc_tmpl":"The {brand} {name} {variant} refrigerator provides optimal food preservation with smart cooling technology and energy efficiency."},

    {"gpc_brick":"10000249","gpc_name":"Ovens & Ranges",
     "google_path":"Home & Garden > Kitchen & Dining > Kitchen Appliances > Ovens",
     "short":"Electronics","yolo":["oven"],"mat":"Electronics","brand":"Electronics",
     "products":["OTG Oven","Air Fryer Oven","Toaster Oven","Smart Oven",
                 "Countertop Oven","Convection Oven","Rotisserie Oven"],
     "models":["12L","19L","25L","28L","30L","38L","40L","Basic","Pro","Digital"],
     "spec_fn":spec_kitchen,
     "desc_tmpl":"The {brand} {name} {variant} oven offers versatile cooking modes including bake, grill, toast, and air-fry for perfect results."},

    {"gpc_brick":"10000250","gpc_name":"Toasters",
     "google_path":"Home & Garden > Kitchen & Dining > Kitchen Appliances > Toasters & Grills > Toasters",
     "short":"Electronics","yolo":["toaster"],"mat":"Electronics","brand":"Electronics",
     "products":["Pop-Up Toaster","Sandwich Maker","Toaster Grill","Bread Toaster"],
     "models":["2-Slice","4-Slice","Stainless","Classic","Digital","Wide Slot","Bagel"],
     "spec_fn":spec_kitchen,
     "desc_tmpl":"The {brand} {name} {variant} toaster delivers perfectly toasted bread every time with adjustable browning control and auto-shutoff safety."},

    {"gpc_brick":"10000251","gpc_name":"Monitors",
     "google_path":"Electronics > Video > Computer Monitors",
     "short":"Electronics","yolo":["tv"],"mat":"Electronics","brand":"Electronics",
     "products":["UltraSharp Monitor","Odyssey Gaming Monitor","ProArt Display",
                 "IPS Monitor","OLED Monitor","Curved Monitor","4K Monitor"],
     "models":["24in","27in","32in","34in","38in","49in","UW","4K","QHD","FHD","144Hz","240Hz"],
     "spec_fn":spec_electronics,
     "desc_tmpl":"The {brand} {name} {variant} monitor offers stunning color accuracy and smooth gameplay with ultra-low latency and HDR support."},

    # ── Apparel & Bags ────────────────────────────────────────
    {"gpc_brick":"10000100","gpc_name":"Backpacks",
     "google_path":"Luggage & Bags > Backpacks",
     "short":"Apparel","yolo":["backpack"],"mat":"Apparel","brand":"Apparel & Accessories",
     "products":["Hiking Backpack","Laptop Backpack","School Bag","Travel Pack",
                 "Daypack","Hydration Pack","Tactical Backpack","Camera Bag"],
     "models":["20L","30L","40L","50L","60L","80L","Mini","Slim","Pro","Classic"],
     "spec_fn":spec_apparel,
     "desc_tmpl":"The {brand} {name} {variant} backpack offers organized storage with padded straps, water resistance, and ergonomic back panel."},

    {"gpc_brick":"10000101","gpc_name":"Handbags",
     "google_path":"Apparel & Accessories > Handbags, Wallets & Cases > Handbags",
     "short":"Apparel","yolo":["handbag"],"mat":"Apparel","brand":"Apparel & Accessories",
     "products":["Tote Bag","Shoulder Bag","Crossbody Bag","Clutch","Satchel",
                 "Hobo Bag","Bucket Bag","Wristlet","Mini Bag","Evening Bag"],
     "models":["Small","Medium","Large","Mini","Classic","Signature","Limited","Premium"],
     "spec_fn":spec_apparel,
     "desc_tmpl":"The {brand} {name} {variant} handbag combines style and functionality with premium stitching, spacious compartments, and durable hardware."},

    {"gpc_brick":"10000102","gpc_name":"Suitcases",
     "google_path":"Luggage & Bags > Suitcases",
     "short":"Apparel","yolo":["suitcase"],"mat":"Apparel","brand":"Apparel & Accessories",
     "products":["Carry-On Luggage","Check-In Luggage","Hardshell Suitcase",
                 "Spinner Luggage","Rolling Trolley","Weekender Bag","Travel Set"],
     "models":["20in","24in","28in","32in","Cabin","Business","Hardside","Softside","4-Wheel"],
     "spec_fn":spec_apparel,
     "desc_tmpl":"The {brand} {name} {variant} suitcase offers 360-degree spinner wheels, TSA-approved locks, and durable shell for worry-free travel."},

    {"gpc_brick":"10000103","gpc_name":"Umbrellas",
     "google_path":"Home & Garden > Parasols & Rain Umbrellas",
     "short":"Apparel","yolo":["umbrella"],"mat":"Apparel","brand":"Apparel & Accessories",
     "products":["Compact Umbrella","Golf Umbrella","Travel Umbrella","Auto Umbrella",
                 "Windproof Umbrella","UV Protection Umbrella","Folding Umbrella"],
     "models":["3-Fold","2-Fold","Stick","Inverted","Auto Open","Double Canopy","Smart"],
     "spec_fn":spec_apparel,
     "desc_tmpl":"The {brand} {name} {variant} umbrella provides reliable protection against rain and UV with its durable canopy and ergonomic handle."},

    {"gpc_brick":"10000104","gpc_name":"Ties",
     "google_path":"Apparel & Accessories > Clothing Accessories > Neckties",
     "short":"Apparel","yolo":["tie"],"mat":"Apparel","brand":"Apparel & Accessories",
     "products":["Silk Tie","Bow Tie","Knit Tie","Striped Tie","Paisley Tie",
                 "Solid Tie","Polka Dot Tie","Clip-On Tie","Slim Tie","Cravat"],
     "models":["Slim 6cm","Standard 8cm","Wide 10cm","Long","Regular","Short"],
     "spec_fn":spec_apparel,
     "desc_tmpl":"The {brand} {name} {variant} tie adds a refined touch to formal and business attire, crafted from premium fabric with precise tailoring."},

    # ── Kitchen & Home ────────────────────────────────────────
    {"gpc_brick":"10000300","gpc_name":"Water Bottles",
     "google_path":"Home & Garden > Kitchen & Dining > Food & Beverage Carriers > Water Bottles",
     "short":"Kitchen","yolo":["bottle"],"mat":"Kitchen","brand":"Kitchen & Home",
     "products":["Insulated Water Bottle","Sports Bottle","Sipper Bottle",
                 "Hydro Flask","Wide Mouth Bottle","Flip-Top Bottle","Tumbler Bottle"],
     "models":["350ml","500ml","600ml","750ml","1L","1.5L","2L"],
     "spec_fn":spec_kitchen,
     "desc_tmpl":"The {brand} {name} {variant} keeps beverages {temp_mode} for hours with vacuum insulation technology and leak-proof lid design."},

    {"gpc_brick":"10000301","gpc_name":"Wine Glasses",
     "google_path":"Home & Garden > Kitchen & Dining > Tableware > Drinkware > Stemware",
     "short":"Kitchen","yolo":["wine glass"],"mat":"Kitchen","brand":"Kitchen & Home",
     "products":["Red Wine Glass","White Wine Glass","Champagne Flute",
                 "Stemless Wine Glass","Crystal Glass","Decanter Set","Wine Carafe"],
     "models":["Set of 2","Set of 4","Set of 6","Single","Lead-Free","Dishwasher Safe"],
     "spec_fn":spec_kitchen,
     "desc_tmpl":"The {brand} {name} {variant} enhances every sip with perfectly balanced crystal clarity and elegant stem design."},

    {"gpc_brick":"10000302","gpc_name":"Cups & Mugs",
     "google_path":"Home & Garden > Kitchen & Dining > Tableware > Drinkware > Mugs",
     "short":"Kitchen","yolo":["cup"],"mat":"Kitchen","brand":"Kitchen & Home",
     "products":["Coffee Mug","Travel Mug","Insulated Cup","Espresso Cup",
                 "Latte Cup","Ceramic Mug","Thermos Cup","Smart Mug"],
     "models":["200ml","250ml","300ml","350ml","400ml","500ml","12oz","16oz"],
     "spec_fn":spec_kitchen,
     "desc_tmpl":"The {brand} {name} {variant} keeps your morning coffee or evening tea at perfect temperature with double-wall insulation."},

    {"gpc_brick":"10000303","gpc_name":"Bowls",
     "google_path":"Home & Garden > Kitchen & Dining > Tableware > Dinnerware > Bowls",
     "short":"Kitchen","yolo":["bowl"],"mat":"Kitchen","brand":"Kitchen & Home",
     "products":["Mixing Bowl","Cereal Bowl","Soup Bowl","Salad Bowl","Serving Bowl",
                 "Ramen Bowl","Fruit Bowl","Storage Bowl","Prep Bowl"],
     "models":["Small","Medium","Large","XL","Set of 3","Set of 6","Stackable","Nesting"],
     "spec_fn":spec_kitchen,
     "desc_tmpl":"The {brand} {name} {variant} bowl is perfect for everyday dining, meal prep, and serving with a durable, food-safe finish."},

    {"gpc_brick":"10000304","gpc_name":"Forks",
     "google_path":"Home & Garden > Kitchen & Dining > Tableware > Flatware > Forks",
     "short":"Kitchen","yolo":["fork"],"mat":"Kitchen","brand":"Kitchen & Home",
     "products":["Dinner Fork","Salad Fork","Dessert Fork","Serving Fork",
                 "Fish Fork","Oyster Fork","Fondue Fork","BBQ Fork"],
     "models":["Set of 6","Set of 12","Single","Polished","Brushed","Mirror Finish"],
     "spec_fn":spec_kitchen,
     "desc_tmpl":"The {brand} {name} {variant} fork is crafted for balance and elegance, perfect for everyday dining or formal table settings."},

    {"gpc_brick":"10000305","gpc_name":"Knives",
     "google_path":"Home & Garden > Kitchen & Dining > Kitchen Tools & Utensils > Kitchen Knives",
     "short":"Kitchen","yolo":["knife"],"mat":"Kitchen","brand":"Kitchen & Home",
     "products":["Chef's Knife","Bread Knife","Santoku Knife","Paring Knife",
                 "Carving Knife","Boning Knife","Utility Knife","Cleaver"],
     "models":["6in","8in","10in","12in","Forged","Stamped","Full Tang","Nakiri","Yanagiba"],
     "spec_fn":spec_kitchen,
     "desc_tmpl":"The {brand} {name} {variant} knife delivers surgical-grade cutting performance with high-carbon steel blade and balanced ergonomic handle."},

    {"gpc_brick":"10000306","gpc_name":"Spoons",
     "google_path":"Home & Garden > Kitchen & Dining > Tableware > Flatware > Spoons",
     "short":"Kitchen","yolo":["spoon"],"mat":"Kitchen","brand":"Kitchen & Home",
     "products":["Dinner Spoon","Soup Spoon","Dessert Spoon","Teaspoon",
                 "Serving Spoon","Ladle","Slotted Spoon","Wooden Spoon"],
     "models":["Set of 6","Set of 12","Single","Polished","Brushed","Bamboo","Silicone"],
     "spec_fn":spec_kitchen,
     "desc_tmpl":"The {brand} {name} {variant} spoon combines functionality and elegance for everyday dining, measuring, and cooking needs."},

    {"gpc_brick":"10000307","gpc_name":"Sinks",
     "google_path":"Hardware > Plumbing > Plumbing Fixtures > Sinks",
     "short":"Kitchen","yolo":["sink"],"mat":"Kitchen","brand":"Kitchen & Home",
     "products":["Kitchen Sink","Bathroom Sink","Farmhouse Sink","Bar Sink",
                 "Undermount Sink","Drop-In Sink","Single Bowl Sink","Double Bowl Sink"],
     "models":["Single","Double","18x16in","24x18in","30x18in","33x22in","SS","Ceramic"],
     "spec_fn":spec_kitchen,
     "desc_tmpl":"The {brand} {name} {variant} sink offers premium construction with easy installation, scratch resistance, and timeless design."},

    {"gpc_brick":"10000308","gpc_name":"Vases",
     "google_path":"Home & Garden > Decor > Vases",
     "short":"Kitchen","yolo":["vase"],"mat":"Default","brand":"Kitchen & Home",
     "products":["Ceramic Vase","Glass Vase","Terracotta Vase","Metal Vase",
                 "Crystal Vase","Bamboo Vase","Marble Vase","Floor Vase"],
     "models":["Small","Medium","Large","Tall","Wide Mouth","Narrow","Set of 3","Decorative"],
     "spec_fn":spec_kitchen,
     "desc_tmpl":"The {brand} {name} {variant} vase adds an elegant focal point to any room with its artisanal craftsmanship and timeless aesthetic."},

    {"gpc_brick":"10000309","gpc_name":"Potted Plants",
     "google_path":"Home & Garden > Plants > Indoor & Outdoor Plants",
     "short":"Kitchen","yolo":["potted plant"],"mat":"Default","brand":"Kitchen & Home",
     "products":["Succulent Pot","Snake Plant","Peace Lily","Money Plant",
                 "ZZ Plant","Monstera","Fiddle Leaf Fig","Cactus Pot","Bonsai"],
     "models":["4in Pot","6in Pot","8in Pot","10in Pot","12in Pot","Hanging","Tabletop","Floor"],
     "spec_fn":spec_kitchen,
     "desc_tmpl":"The {brand} {name} {variant} brings natural beauty and clean air to your home with low-maintenance care requirements."},

    {"gpc_brick":"10000310","gpc_name":"Scissors",
     "google_path":"Arts & Entertainment > Hobbies & Creative Arts > Arts & Crafts > Art & Crafting Tools",
     "short":"Kitchen","yolo":["scissors"],"mat":"Default","brand":"Kitchen & Home",
     "products":["Kitchen Scissors","Craft Scissors","Fabric Scissors","Paper Scissors",
                 "Pruning Shears","Hair Scissors","Office Scissors","Embroidery Scissors"],
     "models":["5in","7in","8in","9in","Left-Handed","Micro-Tip","Stainless","Titanium"],
     "spec_fn":spec_kitchen,
     "desc_tmpl":"The {brand} {name} {variant} scissors offer precision cutting with razor-sharp blades and comfortable ergonomic handle design."},

    {"gpc_brick":"10000311","gpc_name":"Clocks",
     "google_path":"Home & Garden > Decor > Clocks > Wall Clocks",
     "short":"Kitchen","yolo":["clock"],"mat":"Default","brand":"Kitchen & Home",
     "products":["Wall Clock","Alarm Clock","Table Clock","Grandfather Clock",
                 "Digital Clock","LED Clock","Smart Clock","Cuckoo Clock"],
     "models":["10in","12in","14in","18in","Analog","Digital","Silent","Chime","Silent Sweep"],
     "spec_fn":spec_kitchen,
     "desc_tmpl":"The {brand} {name} {variant} clock combines precise timekeeping with stylish design to complement any interior decor."},

    # ── Furniture ─────────────────────────────────────────────
    {"gpc_brick":"10000400","gpc_name":"Chairs",
     "google_path":"Furniture > Chairs > Office Chairs",
     "short":"Furniture","yolo":["chair"],"mat":"Furniture","brand":"Furniture",
     "products":["Office Chair","Gaming Chair","Dining Chair","Accent Chair",
                 "Rocking Chair","Lounge Chair","Bar Stool","Bean Bag Chair"],
     "models":["Ergonomic","Mesh","Executive","Task","Racing","Drafting","Folding","Stackable"],
     "spec_fn":spec_furniture,
     "desc_tmpl":"The {brand} {name} {variant} chair provides exceptional comfort and lumbar support for long working hours with adjustable settings."},

    {"gpc_brick":"10000401","gpc_name":"Sofas & Couches",
     "google_path":"Furniture > Sofas",
     "short":"Furniture","yolo":["couch"],"mat":"Furniture","brand":"Furniture",
     "products":["3-Seater Sofa","L-Shaped Sofa","Sectional Sofa","Loveseat",
                 "Sofa Bed","Futon","Chesterfield Sofa","Recliner Sofa"],
     "models":["2-Seater","3-Seater","4-Seater","L-Shape","U-Shape","Modular","Sleeper","Compact"],
     "spec_fn":spec_furniture,
     "desc_tmpl":"The {brand} {name} {variant} sofa transforms your living room with plush cushioning, durable upholstery, and sophisticated design."},

    {"gpc_brick":"10000402","gpc_name":"Beds",
     "google_path":"Furniture > Beds & Accessories > Beds & Bed Frames",
     "short":"Furniture","yolo":["bed"],"mat":"Furniture","brand":"Furniture",
     "products":["Platform Bed","Storage Bed","Bunk Bed","Divan Bed","Panel Bed",
                 "Murphy Bed","Canopy Bed","Daybed","Adjustable Bed"],
     "models":["Single","Twin","Full","Queen","King","Super King","Twin XL","California King"],
     "spec_fn":spec_furniture,
     "desc_tmpl":"The {brand} {name} {variant} bed frame combines solid construction with elegant design for a restful, stylish bedroom sanctuary."},

    {"gpc_brick":"10000403","gpc_name":"Dining Tables",
     "google_path":"Furniture > Tables > Kitchen & Dining Room Tables",
     "short":"Furniture","yolo":["dining table"],"mat":"Furniture","brand":"Furniture",
     "products":["Dining Table","Extendable Table","Round Table","Oval Table",
                 "Pedestal Table","Farmhouse Table","Glass Top Table","Trestle Table"],
     "models":["2-Seater","4-Seater","6-Seater","8-Seater","Extendable","Foldable","Bar Height"],
     "spec_fn":spec_furniture,
     "desc_tmpl":"The {brand} {name} {variant} dining table creates a perfect gathering space with sturdy construction and timeless design aesthetic."},

    {"gpc_brick":"10000404","gpc_name":"Toilets",
     "google_path":"Hardware > Plumbing > Plumbing Fixtures > Toilets & Bidets > Toilets",
     "short":"Furniture","yolo":["toilet"],"mat":"Default","brand":"Furniture",
     "products":["One-Piece Toilet","Two-Piece Toilet","Wall-Hung Toilet","Smart Toilet",
                 "Corner Toilet","Compact Toilet","Dual Flush Toilet","Bidet Toilet"],
     "models":["Standard","Comfort Height","Elongated","Round","Concealed Tank","Wall Mount"],
     "spec_fn":spec_furniture,
     "desc_tmpl":"The {brand} {name} {variant} toilet combines water efficiency with hygienic design and quiet-close lid for modern bathrooms."},

    {"gpc_brick":"10000405","gpc_name":"Bookcases",
     "google_path":"Furniture > Shelving > Bookcases & Standing Shelves",
     "short":"Furniture","yolo":["book"],"mat":"Furniture","brand":"Furniture",
     "products":["Bookcase","Bookshelf","Open Shelving","Ladder Shelf","Floating Shelf",
                 "Corner Bookshelf","Cube Organizer","Display Cabinet"],
     "models":["3-Shelf","4-Shelf","5-Shelf","6-Shelf","Narrow","Wide","Tall","Low Profile"],
     "spec_fn":spec_furniture,
     "desc_tmpl":"The {brand} {name} {variant} bookcase offers stylish storage for books, decor, and essentials with a sturdy, space-saving design."},

    # ── Tools ─────────────────────────────────────────────────
    {"gpc_brick":"10000500","gpc_name":"Power Drills",
     "google_path":"Hardware > Tools > Drills > Handheld Power Drills",
     "short":"Tools","yolo":["person"],"mat":"Tools","brand":"Hardware & Tools",
     "products":["Cordless Drill","Impact Driver","Hammer Drill","SDS Plus Drill",
                 "Right Angle Drill","Drill Driver","Combi Drill","Rotary Hammer"],
     "models":["12V","18V","20V","36V","Brushless","Compact","Heavy Duty","Kit"],
     "spec_fn":spec_tools,
     "desc_tmpl":"The {brand} {name} {variant} delivers superior drilling power with brushless motor technology and all-day battery performance."},

    {"gpc_brick":"10000501","gpc_name":"Flashlights",
     "google_path":"Hardware > Tools > Flashlights & Headlamps",
     "short":"Tools","yolo":["person"],"mat":"Tools","brand":"Hardware & Tools",
     "products":["LED Flashlight","Tactical Flashlight","Rechargeable Torch",
                 "Headlamp","Work Light","Emergency Torch","Solar Flashlight"],
     "models":["100 Lumens","500 Lumens","1000 Lumens","2000 Lumens","3000 Lumens",
               "Zoomable","Waterproof","EDC","Handheld"],
     "spec_fn":spec_tools,
     "desc_tmpl":"The {brand} {name} {variant} flashlight provides powerful, reliable illumination for outdoor adventures and emergency situations."},

    # ── Sporting Goods ────────────────────────────────────────
    {"gpc_brick":"10000600","gpc_name":"Sports Balls",
     "google_path":"Sporting Goods > Athletics",
     "short":"Sporting","yolo":["sports ball"],"mat":"Sporting","brand":"Sporting Goods",
     "products":["Football","Basketball","Tennis Ball","Baseball","Volleyball",
                 "Cricket Ball","Rugby Ball","Soccer Ball","Golf Ball","Softball"],
     "models":["Size 3","Size 4","Size 5","Training","Match","Pro","Official","Youth"],
     "spec_fn":spec_sporting,
     "desc_tmpl":"The {brand} {name} {variant} is engineered for match play with consistent bounce, durable outer, and precise feel."},

    {"gpc_brick":"10000601","gpc_name":"Tennis Rackets",
     "google_path":"Sporting Goods > Athletics > Tennis > Tennis Racquets",
     "short":"Sporting","yolo":["tennis racket"],"mat":"Sporting","brand":"Sporting Goods",
     "products":["Tennis Racket","Squash Racket","Badminton Racket","Pickleball Paddle",
                 "Table Tennis Paddle","Racketball Racket"],
     "models":["Beginner","Intermediate","Advanced","Pro","Junior","Lightweight","Power","Control"],
     "spec_fn":spec_sporting,
     "desc_tmpl":"The {brand} {name} {variant} racket delivers superior control and power with aerospace-grade frame construction and string bed."},

    {"gpc_brick":"10000602","gpc_name":"Skateboards",
     "google_path":"Sporting Goods > Outdoor Recreation > Skateboarding",
     "short":"Sporting","yolo":["skateboard"],"mat":"Sporting","brand":"Sporting Goods",
     "products":["Complete Skateboard","Longboard","Cruiser Board","Mini Cruiser",
                 "Electric Skateboard","Penny Board","Freestyle Board","Street Deck"],
     "models":["7.5in","7.75in","8.0in","8.25in","8.5in","9in","Complete","Deck Only","Blank"],
     "spec_fn":spec_sporting,
     "desc_tmpl":"The {brand} {name} {variant} skateboard offers smooth rides and precise response with quality trucks, wheels, and bearings."},

    {"gpc_brick":"10000603","gpc_name":"Surfboards",
     "google_path":"Sporting Goods > Outdoor Recreation > Boating & Water Sports > Surfing > Surfboards",
     "short":"Sporting","yolo":["surfboard"],"mat":"Sporting","brand":"Sporting Goods",
     "products":["Shortboard","Longboard Surfboard","Funboard","Fish Surfboard",
                 "Foamboard","Gun Surfboard","Bodyboard","SUP Board"],
     "models":["5ft6","6ft0","6ft4","7ft0","8ft0","9ft0","10ft0","Beginner","Performance"],
     "spec_fn":spec_sporting,
     "desc_tmpl":"The {brand} {name} {variant} surfboard provides excellent glide and maneuverability for surfers of all skill levels."},

    {"gpc_brick":"10000604","gpc_name":"Bicycles",
     "google_path":"Sporting Goods > Outdoor Recreation > Cycling > Bicycles",
     "short":"Sporting","yolo":["bicycle"],"mat":"Sporting","brand":"Sporting Goods",
     "products":["Mountain Bike","Road Bike","Hybrid Bike","Electric Bike","BMX Bike",
                 "Folding Bike","City Bike","Gravel Bike","Kids Bike","Fat Bike"],
     "models":["26in","27.5in","29in","700c","Single Speed","21-Speed","27-Speed","Electric","Carbon"],
     "spec_fn":spec_sporting,
     "desc_tmpl":"The {brand} {name} {variant} bicycle delivers a smooth, confident ride with precision components and durable lightweight frame."},

    {"gpc_brick":"10000605","gpc_name":"Baseball Bats",
     "google_path":"Sporting Goods > Athletics > Baseball & Softball",
     "short":"Sporting","yolo":["baseball bat"],"mat":"Sporting","brand":"Sporting Goods",
     "products":["Aluminum Baseball Bat","Wood Baseball Bat","Composite Bat",
                 "Youth Bat","Slowpitch Bat","Fastpitch Bat","T-Ball Bat","Training Bat"],
     "models":["28in","30in","32in","33in","34in","BBCOR","USA","USSSA","Coaching"],
     "spec_fn":spec_sporting,
     "desc_tmpl":"The {brand} {name} {variant} bat delivers maximum power transfer with perfectly balanced swing weight and optimized barrel diameter."},

    {"gpc_brick":"10000606","gpc_name":"Baseball Gloves",
     "google_path":"Sporting Goods > Athletics > Baseball & Softball > Baseball & Softball Gloves & Mitts",
     "short":"Sporting","yolo":["baseball glove"],"mat":"Sporting","brand":"Sporting Goods",
     "products":["Infield Glove","Outfield Glove","Catcher Mitt","First Base Mitt",
                 "Training Glove","Youth Glove","Softball Glove","T-Ball Glove"],
     "models":["10in","11in","11.5in","12in","12.5in","13in","H-Web","I-Web","Basket Web"],
     "spec_fn":spec_sporting,
     "desc_tmpl":"The {brand} {name} {variant} glove is game-ready with pro-pattern design, premium leather construction, and secure wrist strap."},

    {"gpc_brick":"10000607","gpc_name":"Frisbees",
     "google_path":"Sporting Goods > Outdoor Recreation > Outdoor Games",
     "short":"Sporting","yolo":["frisbee"],"mat":"Sporting","brand":"Sporting Goods",
     "products":["Ultimate Frisbee","Disc Golf Disc","Freestyle Disc",
                 "Dog Frisbee","Beach Frisbee","LED Frisbee","Mini Frisbee"],
     "models":["Standard 175g","Ultimate 175g","Lightweight","Heavy","Distance","Putt & Approach"],
     "spec_fn":spec_sporting,
     "desc_tmpl":"The {brand} {name} {variant} disc delivers long, accurate flight with excellent stability and comfortable grip in all conditions."},

    {"gpc_brick":"10000608","gpc_name":"Skis",
     "google_path":"Sporting Goods > Outdoor Recreation > Winter Sports & Activities > Skiing & Snowboarding > Skis",
     "short":"Sporting","yolo":["skis"],"mat":"Sporting","brand":"Sporting Goods",
     "products":["Downhill Skis","Cross-Country Skis","Freestyle Skis","Powder Skis",
                 "Carving Skis","All-Mountain Skis","Racing Skis","Park Skis"],
     "models":["150cm","160cm","165cm","170cm","175cm","180cm","Beginner","Expert","Powder","Race"],
     "spec_fn":spec_sporting,
     "desc_tmpl":"The {brand} {name} {variant} skis provide exceptional edge control and stability on groomed runs and off-piste terrain alike."},

    {"gpc_brick":"10000609","gpc_name":"Snowboards",
     "google_path":"Sporting Goods > Outdoor Recreation > Winter Sports & Activities > Skiing & Snowboarding > Snowboards",
     "short":"Sporting","yolo":["snowboard"],"mat":"Sporting","brand":"Sporting Goods",
     "products":["Freestyle Snowboard","All-Mountain Snowboard","Freeride Board",
                 "Carving Snowboard","Powder Snowboard","Split Board","Park Board"],
     "models":["148cm","152cm","155cm","158cm","162cm","165cm","Twin","Directional","Tapered"],
     "spec_fn":spec_sporting,
     "desc_tmpl":"The {brand} {name} {variant} snowboard handles any terrain with responsive flex, quality base material, and precise edge control."},

    {"gpc_brick":"10000610","gpc_name":"Kites",
     "google_path":"Toys & Games > Toys > Flying Toys > Kites",
     "short":"Sporting","yolo":["kite"],"mat":"Sporting","brand":"Sporting Goods",
     "products":["Delta Kite","Diamond Kite","Box Kite","Stunt Kite","Power Kite",
                 "Fighter Kite","Kids Kite","Beach Kite","LED Kite"],
     "models":["Small","Medium","Large","XL","Beginner","Professional","Dual-Line","Quad-Line"],
     "spec_fn":spec_sporting,
     "desc_tmpl":"The {brand} {name} {variant} kite soars effortlessly in wind with a vibrant ripstop nylon canopy and fiberglass frame."},

    # ── Food & Beverage ───────────────────────────────────────
    {"gpc_brick":"10000700","gpc_name":"Bananas",
     "google_path":"Food, Beverages & Tobacco > Food Items > Fruits & Vegetables > Fresh & Frozen Fruits",
     "short":"Food","yolo":["banana"],"mat":"Default","brand":"Food & Beverage",
     "products":["Cavendish Banana","Plantain","Baby Banana","Red Banana",
                 "Organic Banana","Dried Banana","Banana Chips"],
     "models":["Single","Bunch","500g","1kg","2kg","Organic","Conventional","Fair Trade"],
     "spec_fn":spec_food,
     "desc_tmpl":"The {brand} {name} {variant} delivers natural sweetness, essential potassium, and vitamins for a healthy everyday snack."},

    {"gpc_brick":"10000701","gpc_name":"Apples",
     "google_path":"Food, Beverages & Tobacco > Food Items > Fruits & Vegetables > Fresh & Frozen Fruits > Apples",
     "short":"Food","yolo":["apple"],"mat":"Default","brand":"Food & Beverage",
     "products":["Fuji Apple","Gala Apple","Granny Smith","Honeycrisp","Braeburn",
                 "Pink Lady","Golden Delicious","Jazz Apple","Cosmic Crisp"],
     "models":["Single","3-Pack","6-Pack","500g","1kg","2kg","Organic","Premium","Gift Box"],
     "spec_fn":spec_food,
     "desc_tmpl":"The {brand} {name} {variant} apple offers crisp texture, natural sweetness, and rich Vitamin C for daily nutrition and freshness."},

    {"gpc_brick":"10000702","gpc_name":"Sandwiches",
     "google_path":"Food, Beverages & Tobacco > Food Items > Prepared Foods",
     "short":"Food","yolo":["sandwich"],"mat":"Default","brand":"Food & Beverage",
     "products":["Club Sandwich","BLT Sandwich","Grilled Cheese","Veggie Wrap",
                 "Panini","Sub Sandwich","Breakfast Sandwich","Chicken Wrap"],
     "models":["Regular","Large","Combo","Meal Deal","Vegan","Gluten-Free","Low Cal"],
     "spec_fn":spec_food,
     "desc_tmpl":"The {brand} {name} {variant} sandwich is made with fresh, premium ingredients and artisan bread for a satisfying, flavorful meal."},

    {"gpc_brick":"10000703","gpc_name":"Oranges",
     "google_path":"Food, Beverages & Tobacco > Food Items > Fruits & Vegetables > Fresh & Frozen Fruits > Citrus Fruits > Oranges",
     "short":"Food","yolo":["orange"],"mat":"Default","brand":"Food & Beverage",
     "products":["Navel Orange","Valencia Orange","Blood Orange","Mandarin",
                 "Clementine","Cara Cara Orange","Organic Orange"],
     "models":["Single","3-Pack","6-Pack","1kg","2kg","Organic","Premium","Juicing"],
     "spec_fn":spec_food,
     "desc_tmpl":"The {brand} {name} {variant} orange is bursting with natural citrus flavor and Vitamin C for immune support and refreshment."},

    {"gpc_brick":"10000704","gpc_name":"Broccoli",
     "google_path":"Food, Beverages & Tobacco > Food Items > Fruits & Vegetables > Fresh & Frozen Vegetables > Broccoli",
     "short":"Food","yolo":["broccoli"],"mat":"Default","brand":"Food & Beverage",
     "products":["Fresh Broccoli","Organic Broccoli","Tenderstem Broccoli",
                 "Frozen Broccoli","Baby Broccoli","Broccolini"],
     "models":["Single Head","500g","1kg","Florets","Organic","Steam Bag","Fresh Cut"],
     "spec_fn":spec_food,
     "desc_tmpl":"The {brand} {name} {variant} broccoli is farm-fresh, nutrient-dense and perfect for steaming, roasting, or stir-frying."},

    {"gpc_brick":"10000705","gpc_name":"Carrots",
     "google_path":"Food, Beverages & Tobacco > Food Items > Fruits & Vegetables > Fresh & Frozen Vegetables > Carrots",
     "short":"Food","yolo":["carrot"],"mat":"Default","brand":"Food & Beverage",
     "products":["Baby Carrots","Whole Carrots","Organic Carrots","Rainbow Carrots",
                 "Frozen Carrots","Shredded Carrots","Juicing Carrots"],
     "models":["250g","500g","1kg","2kg","Organic","Snack Pack","Julienne","Mini"],
     "spec_fn":spec_food,
     "desc_tmpl":"The {brand} {name} {variant} carrots are sweet, crunchy, and rich in beta-carotene for healthy eyes and immune function."},

    {"gpc_brick":"10000706","gpc_name":"Hot Dogs",
     "google_path":"Food, Beverages & Tobacco > Food Items > Prepared Foods > Prepared Meals & Entrees",
     "short":"Food","yolo":["hot dog"],"mat":"Default","brand":"Food & Beverage",
     "products":["Classic Hot Dog","Beef Hot Dog","Chicken Frank","Veggie Dog",
                 "Jumbo Frank","Mini Hot Dog","Gourmet Dog","Corn Dog"],
     "models":["Single","4-Pack","6-Pack","8-Pack","Family Pack","Jumbo","Regular","Skinless"],
     "spec_fn":spec_food,
     "desc_tmpl":"The {brand} {name} {variant} is a delicious, ready-to-grill classic made with premium cuts and natural seasonings."},

    {"gpc_brick":"10000707","gpc_name":"Pizza",
     "google_path":"Food, Beverages & Tobacco > Food Items > Prepared Foods",
     "short":"Food","yolo":["pizza"],"mat":"Default","brand":"Food & Beverage",
     "products":["Margherita Pizza","Pepperoni Pizza","BBQ Chicken Pizza",
                 "Veggie Pizza","Four Cheese Pizza","Hawaiian Pizza","Thin Crust Pizza"],
     "models":["Personal 6in","Small 9in","Medium 12in","Large 14in","XL 16in",
               "Frozen","Take-Away","Gourmet","Vegan","Gluten-Free"],
     "spec_fn":spec_food,
     "desc_tmpl":"The {brand} {name} {variant} pizza is crafted with authentic hand-stretched dough, rich tomato sauce, and premium toppings."},

    {"gpc_brick":"10000708","gpc_name":"Donuts",
     "google_path":"Food, Beverages & Tobacco > Food Items > Bakery > Donuts",
     "short":"Food","yolo":["donut"],"mat":"Default","brand":"Food & Beverage",
     "products":["Glazed Donut","Chocolate Donut","Sprinkle Donut","Filled Donut",
                 "Old Fashioned Donut","Cruller","Cronut","Vegan Donut"],
     "models":["Single","6-Pack","12-Box","Assorted","Glazed","Frosted","Cake","Yeast"],
     "spec_fn":spec_food,
     "desc_tmpl":"The {brand} {name} {variant} donut is freshly baked with premium ingredients for the perfect indulgent treat any time of day."},

    {"gpc_brick":"10000709","gpc_name":"Cakes",
     "google_path":"Food, Beverages & Tobacco > Food Items > Bakery > Cakes & Dessert Bars",
     "short":"Food","yolo":["cake"],"mat":"Default","brand":"Food & Beverage",
     "products":["Chocolate Cake","Vanilla Cake","Red Velvet","Cheesecake","Carrot Cake",
                 "Lemon Drizzle","Black Forest","Tiramisu","Chiffon Cake","Pound Cake"],
     "models":["6in","8in","10in","12in","Slice","Mini","Birthday","Wedding","Gluten-Free","Vegan"],
     "spec_fn":spec_food,
     "desc_tmpl":"The {brand} {name} {variant} cake is lovingly baked with premium ingredients for celebrations and everyday sweet indulgences."},

    # ── Toys & Misc ───────────────────────────────────────────
    {"gpc_brick":"10000800","gpc_name":"Teddy Bears",
     "google_path":"Toys & Games > Toys > Dolls, Playsets & Toy Figures > Stuffed Animals",
     "short":"Toys","yolo":["teddy bear"],"mat":"Apparel","brand":"Toys & Games",
     "products":["Classic Teddy Bear","Giant Teddy Bear","Stuffed Elephant",
                 "Plush Bunny","Stuffed Panda","Plush Lion","Baby Teddy","Glow Teddy"],
     "models":["6in","10in","12in","18in","24in","36in","48in","Jumbo","Mini","Talking"],
     "spec_fn":spec_toy,
     "desc_tmpl":"The {brand} {name} {variant} is the perfect cuddly companion crafted from ultra-soft plush fabric for hugs and imaginative play."},

    {"gpc_brick":"10000801","gpc_name":"Hair Dryers",
     "google_path":"Health & Beauty > Personal Care > Hair Care > Hair Styling Tools > Hair Dryers",
     "short":"Electronics","yolo":["hair drier"],"mat":"Electronics","brand":"Electronics",
     "products":["Ionic Hair Dryer","Tourmaline Dryer","Travel Hair Dryer",
                 "Diffuser Dryer","Smart Hair Dryer","Salon Professional Dryer"],
     "models":["1200W","1600W","1800W","2000W","2400W","Compact","Folding","Lightweight"],
     "spec_fn":spec_electronics,
     "desc_tmpl":"The {brand} {name} {variant} hair dryer delivers fast drying with ionic technology for frizz-free, salon-quality results at home."},

    {"gpc_brick":"10000802","gpc_name":"Toothbrushes",
     "google_path":"Health & Beauty > Personal Care > Oral Care > Toothbrushes",
     "short":"Toys","yolo":["toothbrush"],"mat":"Default","brand":"Baby & Toddler",
     "products":["Electric Toothbrush","Sonic Toothbrush","Manual Toothbrush",
                 "Bamboo Toothbrush","Kids Toothbrush","Travel Toothbrush","Smart Toothbrush"],
     "models":["Soft","Medium","Hard","Ultra Soft","Kids","Premium","Standard","Replacement Head"],
     "spec_fn":spec_toy,
     "desc_tmpl":"The {brand} {name} {variant} toothbrush ensures thorough plaque removal with advanced bristle technology and comfortable grip handle."},

    # ── Vehicles ──────────────────────────────────────────────
    {"gpc_brick":"10000A01","gpc_name":"Cars",
     "google_path":"Vehicles & Parts > Vehicles > Motor Vehicles > Cars, Trucks & Vans",
     "short":"Vehicles","yolo":["car"],"mat":"Default","brand":"Vehicles",
     "products":["Sedan","SUV","Hatchback","Compact Car","Crossover","Station Wagon",
                 "Electric Car","Hybrid Car","Sports Car","Coupe"],
     "models":["Entry","Standard","Premium","Sport","Limited","Touring","GT","EV","Hybrid"],
     "spec_fn":spec_vehicle,
     "desc_tmpl":"The {brand} {name} {variant} delivers refined performance, exceptional safety ratings, and class-leading fuel efficiency."},

    {"gpc_brick":"10000A02","gpc_name":"Motorcycles",
     "google_path":"Vehicles & Parts > Vehicles > Motor Vehicles > Motorcycles & Scooters",
     "short":"Vehicles","yolo":["motorcycle"],"mat":"Default","brand":"Vehicles",
     "products":["Sport Bike","Cruiser","Adventure Bike","Naked Bike","Touring Bike",
                 "Scooter","Electric Scooter","Dirt Bike","Commuter Bike","Cafe Racer"],
     "models":["100cc","125cc","150cc","200cc","250cc","400cc","600cc","800cc","1000cc","EV"],
     "spec_fn":spec_vehicle,
     "desc_tmpl":"The {brand} {name} {variant} motorcycle combines thrilling performance with urban agility and premium safety features."},

    {"gpc_brick":"10000A03","gpc_name":"Trucks",
     "google_path":"Vehicles & Parts > Vehicles > Motor Vehicles > Cars, Trucks & Vans",
     "short":"Vehicles","yolo":["truck"],"mat":"Default","brand":"Vehicles",
     "products":["Pickup Truck","Light Duty Truck","Heavy Duty Truck","Flatbed Truck",
                 "Dump Truck","Utility Truck","Box Truck","Crew Cab","Extended Cab"],
     "models":["Half-Ton","Three-Quarter-Ton","1-Ton","4x4","4x2","Diesel","V8","Electric","Hybrid"],
     "spec_fn":spec_vehicle,
     "desc_tmpl":"The {brand} {name} {variant} truck combines rugged capability with refined interior and towing prowess for work and adventure."},

    {"gpc_brick":"10000A04","gpc_name":"Buses",
     "google_path":"Vehicles & Parts > Vehicles > Motor Vehicles",
     "short":"Vehicles","yolo":["bus"],"mat":"Default","brand":"Vehicles",
     "products":["City Bus","School Bus","Coach Bus","Mini Bus","Double Decker",
                 "Electric Bus","Shuttle Bus","Tourist Bus","Articulated Bus"],
     "models":["Small 20-seat","Medium 35-seat","Standard 50-seat","Large 70-seat",
               "Diesel","CNG","Electric","Hybrid"],
     "spec_fn":spec_vehicle,
     "desc_tmpl":"The {brand} {name} {variant} provides comfortable, efficient mass transportation with advanced safety systems."},

    {"gpc_brick":"10000A05","gpc_name":"Trains",
     "google_path":"Arts & Entertainment > Hobbies & Creative Arts > Model Making > Model Trains & Train Sets",
     "short":"Vehicles","yolo":["train"],"mat":"Default","brand":"Vehicles",
     "products":["Electric Train Set","Bullet Train Model","Steam Engine Model",
                 "Freight Train Set","Passenger Train","Metro Model","Monorail Model"],
     "models":["N Scale","HO Scale","O Scale","G Scale","Beginner Set","Expert","DCC","Analog"],
     "spec_fn":spec_vehicle,
     "desc_tmpl":"The {brand} {name} {variant} train set delivers realistic details, smooth operation, and expandable track system for enthusiasts."},

    {"gpc_brick":"10000A06","gpc_name":"Airplanes",
     "google_path":"Vehicles & Parts > Vehicle Parts & Accessories > Aircraft Parts & Accessories",
     "short":"Vehicles","yolo":["airplane"],"mat":"Default","brand":"Vehicles",
     "products":["RC Airplane","Glider Model","Propeller Plane Model",
                 "Jet Fighter Model","Commercial Jet Model","Biplane Model"],
     "models":["Beginner RTF","Intermediate","Advanced","Scale 1:72","Scale 1:48","ARF","BNF"],
     "spec_fn":spec_vehicle,
     "desc_tmpl":"The {brand} {name} {variant} airplane model offers exceptional aerodynamic stability and realistic detail for display or flight."},

    {"gpc_brick":"10000A07","gpc_name":"Boats",
     "google_path":"Sporting Goods > Outdoor Recreation > Boating & Water Sports > Boating & Rafting",
     "short":"Vehicles","yolo":["boat"],"mat":"Default","brand":"Vehicles",
     "products":["RC Speedboat","Kayak","Canoe","Inflatable Raft","Rowboat","Sailboat",
                 "Motorboat Model","Pontoon Model","Fishing Boat","Jon Boat"],
     "models":["Beginner","Advanced","Scale 1:10","Single","Tandem","Electric","Gas","Manual"],
     "spec_fn":spec_vehicle,
     "desc_tmpl":"The {brand} {name} {variant} boat delivers smooth performance on water with responsive handling and durable hull construction."},

    {"gpc_brick":"10000A08","gpc_name":"Bench",
     "google_path":"Furniture > Benches",
     "short":"Furniture","yolo":["bench"],"mat":"Furniture","brand":"Furniture",
     "products":["Garden Bench","Entryway Bench","Storage Bench","Weight Bench",
                 "Park Bench","Picnic Bench","Shoe Bench","Bedroom Bench"],
     "models":["2-Seater","3-Seater","4-Seater","Folding","Padded","Cushioned","Metal","Wood"],
     "spec_fn":spec_furniture,
     "desc_tmpl":"The {brand} {name} {variant} bench provides comfortable seating with sturdy construction and weather-resistant finish."},

    {"gpc_brick":"10000B01","gpc_name":"Traffic Lights",
     "google_path":"Business & Industrial > Signage > Road & Traffic Signs",
     "short":"Electronics","yolo":["traffic light"],"mat":"Tools","brand":"Hardware & Tools",
     "products":["LED Traffic Light","Portable Traffic Signal","Pedestrian Signal",
                 "Solar Traffic Light","Countdown Timer Signal","School Zone Signal"],
     "models":["12in LED","16in LED","20in LED","Solar Powered","Wireless","Wired","3-Phase"],
     "spec_fn":spec_tools,
     "desc_tmpl":"The {brand} {name} {variant} traffic signal uses energy-efficient LEDs for maximum visibility and minimal maintenance requirements."},

    {"gpc_brick":"10000B02","gpc_name":"Fire Hydrants",
     "google_path":"Home & Garden > Flood, Fire & Gas Safety",
     "short":"Tools","yolo":["fire hydrant"],"mat":"Tools","brand":"Hardware & Tools",
     "products":["Wet Barrel Hydrant","Dry Barrel Hydrant","Yard Hydrant",
                 "Wall Hydrant","Post Hydrant","Flush Hydrant","Gate Valve Hydrant"],
     "models":["4.5in","6in","8in","AWWA","UL-FM Approved","Standard","High Flow","Low Pressure"],
     "spec_fn":spec_tools,
     "desc_tmpl":"The {brand} {name} {variant} fire hydrant meets AWWA standards for reliable high-flow water access during emergency response."},

    {"gpc_brick":"10000B03","gpc_name":"Stop Signs",
     "google_path":"Business & Industrial > Signage > Road & Traffic Signs",
     "short":"Tools","yolo":["stop sign"],"mat":"Default","brand":"Hardware & Tools",
     "products":["Reflective Stop Sign","Portable Stop Sign","LED Stop Sign",
                 "Mini Stop Sign","Temporary Stop Sign","Mounted Stop Sign"],
     "models":["12in","18in","24in","30in","36in","Engineer Grade","Diamond Grade","High Intensity"],
     "spec_fn":spec_tools,
     "desc_tmpl":"The {brand} {name} {variant} stop sign meets MUTCD standards with high-visibility retroreflective sheeting for maximum safety."},

    {"gpc_brick":"10000B04","gpc_name":"Parking Meters",
     "google_path":"Business & Industrial > Signage > Parking Signs & Permits",
     "short":"Electronics","yolo":["parking meter"],"mat":"Tools","brand":"Hardware & Tools",
     "products":["Single Space Meter","Multi-Space Pay Station","Solar Parking Meter",
                 "Smart Parking Meter","Contactless Pay Meter","Digital Parking Meter"],
     "models":["Coin-Only","Card & Coin","Contactless","Solar","Networked","Stand-Alone","Kiosk"],
     "spec_fn":spec_tools,
     "desc_tmpl":"The {brand} {name} {variant} parking meter provides reliable pay-to-park management with tamper-resistant design and remote monitoring."},

    {"gpc_brick":"10000B05","gpc_name":"Books",
     "google_path":"Media > Books > Print Books",
     "short":"Default","yolo":["book"],"mat":"Default","brand":"Food & Beverage",
     "products":["Hardcover Novel","Paperback Novel","Textbook","Cookbook",
                 "Self-Help Book","Children's Book","Comic Book","Encyclopedia",
                 "Reference Book","Art Book","Biography","Poetry Collection"],
     "models":["Hardcover","Paperback","Large Print","Collector's Edition",
               "Anniversary Edition","Illustrated","Pocket","Annotated","Special Edition"],
     "spec_fn":spec_default,
     "desc_tmpl":"The {brand} {name} {variant} is a must-read that captivates with compelling narrative, rich detail, and enduring relevance."},

    {"gpc_brick":"10000B06","gpc_name":"Person & Athletic Gear",
     "google_path":"Sporting Goods > Athletics",
     "short":"Default","yolo":["person"],"mat":"Default","brand":"Sporting Goods",
     "products":["Running Shoes","Gym Gloves","Athletic Shorts","Compression Tights",
                 "Sport Jersey","Cycling Helmet","Swim Goggles","Boxing Gloves",
                 "Weight Belt","Knee Brace","Ankle Support","Wrist Guard"],
     "models":["XS","S","M","L","XL","XXL","Youth","Adult","Pro","Standard"],
     "spec_fn":spec_sporting,
     "desc_tmpl":"The {brand} {name} {variant} is engineered for peak athletic performance, providing comfort, support, and durability in action."},

    {"gpc_brick":"10000B07","gpc_name":"Birds & Bird Products",
     "google_path":"Animals & Pet Supplies > Pet Supplies > Bird Supplies",
     "short":"Default","yolo":["bird"],"mat":"Default","brand":"Baby & Toddler",
     "products":["Plush Bird Toy","Bird Feeder","Bird Bath","Bird Cage","Bird House",
                 "Decorative Bird Sculpture","Wind Chime Bird","Ceramic Bird"],
     "models":["Small","Medium","Large","Indoor","Outdoor","Decorative","Functional","Set"],
     "spec_fn":spec_default,
     "desc_tmpl":"The {brand} {name} {variant} brings a touch of nature to your garden or home with vibrant design and quality craftsmanship."},
]

# ──────────────────────────────────────────────────────────────
# 7. Logo description templates
# ──────────────────────────────────────────────────────────────
def make_logo_desc(brand: str) -> str:
    styles = [
        f"{brand} wordmark in bold sans-serif font on a {random.choice(['white','black','transparent'])} background",
        f"{brand} icon logo - minimalist {random.choice(['circular','shield','hexagonal','square'])} emblem",
        f"{brand} combination mark with stylized letterform and product category text",
        f"{brand} brand seal with registered trademark symbol, monochromatic",
        f"{brand} logo with gradient accent on primary brand color",
        f"Embossed {brand} logo on product surface, single-color imprint",
        f"{brand} stacked logo variant: icon above name in medium-weight type",
        f"{brand} logo on packaging - reverse white-out on brand primary color",
    ]
    return random.choice(styles)

# ──────────────────────────────────────────────────────────────
# 8. Helpers
# ──────────────────────────────────────────────────────────────
def get_material(mat_key: str) -> str:
    key = mat_key if mat_key in MATERIALS else "Default"
    return random.choice(MATERIALS[key])

def get_brand(brand_key: str) -> str:
    return random.choice(BRANDS.get(brand_key, ALL_BRANDS))

def fake_gtin() -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(13))

def fake_mpn(brand: str, model: str) -> str:
    prefix = "".join(c for c in brand.upper() if c.isalpha())[:3]
    suffix = hashlib.md5(f"{brand}{model}{random.random()}".encode()).hexdigest()[:6].upper()
    return f"{prefix}-{suffix}"

# ──────────────────────────────────────────────────────────────
# 9. Row generator
# ──────────────────────────────────────────────────────────────
VARIANTS = ["Gen 1","Gen 2","Gen 3","V2","V3","2023 Edition","2024 Edition",
            "Limited","Pro Edition","Special Edition","Anniversary","Classic",
            "New","Upgraded","Refreshed","Revised","Signature","Exclusive",
            "Premium","Value Pack","Bundle","Combo","Starter","Deluxe"]

RES_OPTS   = ["FHD","QHD","4K","8K","OLED","AMOLED","LCD"]
TEMP_MODES = ["cold","hot","warm","ice-cold","piping hot"]

def generate_rows(n: int = 10_000) -> list:
    rows     = []
    seen_keys = set()
    attempts  = 0
    max_att   = n * 15

    while len(rows) < n and attempts < max_att:
        attempts += 1
        cat     = random.choice(CATEGORIES)
        product = random.choice(cat["products"])
        model   = random.choice(cat["models"])
        brand   = get_brand(cat["brand"])
        variant = random.choice(VARIANTS)

        uk = f"{brand}|{product}|{model}|{variant}"
        if uk in seen_keys:
            continue
        seen_keys.add(uk)

        yolo_name    = random.choice(cat["yolo"])
        display_name = f"{brand} {product} {model} {variant}"
        color        = random.choice(COLORS)
        material     = get_material(cat["mat"])
        logo_desc    = make_logo_desc(brand)
        gtin         = fake_gtin()
        mpn          = fake_mpn(brand, model)
        specs_raw    = cat["spec_fn"](brand, product, variant)

        full_specs = (
            f"{specs_raw} | Color:{color} | Material:{material} | "
            f"GTIN:{gtin} | MPN:{mpn} | "
            f"GS1_Brick:{cat['gpc_brick']} ({cat['gpc_name']}) | "
            f"BTG_Path:{cat['google_path']}"
        )

        desc = (cat["desc_tmpl"]
                .format(brand=brand, name=product, variant=variant,
                        cat=cat["gpc_name"],
                        res=random.choice(RES_OPTS),
                        temp_mode=random.choice(TEMP_MODES)))

        rows.append({
            "yolo_name"   : yolo_name,
            "display_name": display_name,
            "category"    : cat["google_path"],
            "color"       : color,
            "material"    : material,
            "brand"       : brand,
            "logo_desc"   : logo_desc,
            "specs"       : full_specs,
            "description" : desc,
            "confirmed"   : random.choice([True, False, False]),
        })

    return rows

# ──────────────────────────────────────────────────────────────
# 10. Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_DIR  = "raw_data_folder"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "global_object_dataset.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Global Object Dataset Generator")
    print("=" * 60)
    print(f"  Categories loaded : {len(CATEGORIES)}")
    print(f"  YOLO classes      : {len(YOLO_CLASSES)}")
    print(f"  Brand pool        : {len(ALL_BRANDS)}")
    print(f"  Color options     : {len(COLORS)}")
    print(f"  Target records    : 10,000")
    print("=" * 60)
    print("Generating rows...")

    data = generate_rows(10_000)
    df   = pd.DataFrame(data)

    yolo_dist = df["yolo_name"].value_counts()
    cat_dist  = df["category"].apply(lambda x: x.split(" > ")[0]).value_counts()

    print(f"\nRecords generated        : {len(df):,}")
    print(f"Unique display_names     : {df['display_name'].nunique():,}")
    print(f"Unique brands            : {df['brand'].nunique():,}")
    print(f"YOLO classes covered     : {df['yolo_name'].nunique():,} / 80")
    print(f"Unique Google categories : {df['category'].nunique():,}")
    print(f"Confirmed (True)         : {df['confirmed'].sum():,}")
    print()
    print("── Top 15 YOLO classes ──────────────────────────────")
    print(yolo_dist.head(15).to_string())
    print()
    print("── Google Taxonomy L1 breakdown ─────────────────────")
    print(cat_dist.to_string())

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print()
    print(f"Saved  ->  {OUTPUT_FILE}")
    sz = os.path.getsize(OUTPUT_FILE)
    print(f"File size : {sz/1024:.1f} KB  ({sz/1024/1024:.2f} MB)")
    print("=" * 60)
    print("Done! Import with:  pd.read_csv('raw_data_folder/global_object_dataset.csv')")
