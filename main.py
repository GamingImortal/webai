import sys
from pathlib import Path
from typing import Dict
import asyncio  # ADDED: For retry delays

def ensure_virtual_environment() -> None:
    if sys.prefix == sys.base_prefix:
        raise SystemExit(
            "This app must run inside a virtual environment. "
            "Use .\\run.ps1 from the project root."
        )

ensure_virtual_environment()

import os
import re
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    Request,
    HTTPException,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from dotenv import load_dotenv
import json
import numpy as np

# optional retrieval deps
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except Exception:
    faiss = None
    SentenceTransformer = None

load_dotenv()

# Debug: Print Ollama configuration
print(f"🔧 Ollama Config:")
print(f"  - BASE_URL: {os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')}")
print(f"  - MODEL: {os.getenv('OLLAMA_MODEL', 'Not set (will auto-detect)')}")

# Test connection to Ollama on startup
try:
    import httpx
    test_response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5.0)
    if test_response.status_code == 200:
        models = test_response.json().get("models", [])
        print(f"✅ Ollama is running with {len(models)} models:")
        for m in models:
            print(f"   - {m.get('name')}")
    else:
        print(f"⚠️ Ollama returned status: {test_response.status_code}")
except Exception as e:
    print(f"❌ Cannot connect to Ollama: {e}")
    print("   Make sure Ollama is running with 'ollama serve'")

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
_AUTO_OLLAMA_MODEL: str | None = None

PREFERRED_OLLAMA_MODELS = (
    "llama3.2:3b",
    "llama3.2",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# SIRD DATASET - ADD ALL YOUR DATA HERE
# ============================================
SIRD_DOCUMENTS = [
    {
        "text": """
SIRDC stands for the Scientific and Industrial Research and Development Centre.
The Scientific and Industrial Research and Development Centre (SIRDC) is a Zimbabwean government-established institution founded in February 1993 under the Research Act [Chapter 10:22] to promote research and development for national industrial, agricultural, and service sectors 
sirdc.ac.zw
sirdc.ac.zw
. Its mandate includes carrying out strategic R&D, adapting imported technologies to local needs, providing consultancy services, and commercializing research outputs through initiatives like Sirtech Investments (Pvt) Ltd
       
        
          """,
       
    },
    {
        "text": """
        https://sirdc.ac.zw/category/courses-on-offer/
        """,
    },
    
    
    
    {"text":"""# AFRICA-UniNet

## About AFRICA-UniNET

- Africa-UniNet is an Austrian-African research network established to promote long-term collaboration between universities and research institutions in Austria and Africa.
- It is funded by the Austrian Federal Ministry of Women's Affairs, Science and Research (BMFWF).
- The network fosters scientific partnerships, innovative joint research projects, and knowledge exchange aligned with the Sustainable Development Goals (SDGs).
- Africa-UniNet has over 70 active members across 17 African countries.
- It provides funding opportunities for collaborative research projects involving Austrian and African institutions.
- The network offers capacity-building programs, including exchange visits, workshops, and further education measures.
- Africa-UniNet serves as a platform for academic cooperation and networking to strengthen research impact and international collaboration.
- It supports inclusive market system development, healthy diets, sustainable consumption, and policy influencing in research programs.
- The network encourages projects that contribute to climate resilience, sustainability, and socio-economic development.
- Governance includes a General Assembly with country representatives, and a Board consisting of Austrian and African members.
- Africa-UniNet organizes calls for project proposals, typically funding research-based, cross-continental collaborations for up to two years with grants up to 40,000 euros.
# Agriculture Cluster

## Overview

The cluster pioneers agricultural transformation through cutting-edge Research and Development in plant and livestock.

## Bioprocessing and Value Addition

- Development of low-cost pre- and postharvest handling systems.
- Support for processing and value addition of crop products.

## Climate Resilience and Sustainability

- Promotion of climate-smart agriculture.
- Integrated Pest and Water Management practices.

## Crop production and improvement

Development of high-yielding crop varieties adapted to various agro ecological zones.
- Production of disease-free seed potato using tissue culture and aeroponics
- Enhancement of underutilized crops such as sweet potato
- Support for rice seed systems with superior upland varieties
- Oyster and Button
- Vegetable and herbal nursery

## Livestock Production

Advisory services on good husbandry practices in:
- Cattle
- Goat
- Sheep and
- Poultry production systems

## Products

- Foundation Seed for Maize, Sunflower, Sorghum, Pearl Millet, Finger Millet, Sugar beans, Wheat, Chia and rice.
- Virus free sweet potato vines
- Diamond seed potato
- Dried herbs of basil, lemon grass, nettle, oregano, peppermint, pennywort, rosemary, thyme, yarrow, and turmeric
- Potted Herb plants basil, lemon grass, nettle, oregano, peppermint, pennywort, rosemary, thyme, and yarrow.
- Mushroom spawn for Oyster and button Mushroom
- Fresh oyster mushrooms
- Dried oyster mushrooms
- Fresh button mushrooms
- Dried button mushrooms
- Oyster mushroom kits
- Indigenous chickens (Burf Orpington, Light Sussex, Black Australorp, Potchefstroom Koekoek)
- Black Solder Fly larva

## Services

- Tissue culture services
- Interpretation of soil analysis data and recommendations
- Molecular biology services (PCR, Gel Electrophoresis, DNA and RNA purification, Incubation, and Culturing services)
- Hatchery Services
- Foundation seed and potato seed production.

## Soil and Nutrient Management

- Development of nutrient management frameworks.
- Promotion of Integrated Nutrient Management (INM) strategies.

## Training and Advisory Services

Capacity building in:
- Good Agricultural Practices
- Crop and livestock husbandry.
- Dissemination of biotechnology innovations.
- Consultancy with local andinternational partners.
# Board Member Profiles

## Overview

Misheck Samson Kachere, Board Chairperson
- Industrialist and business executive
- Degrees named on site: BSc in Chemistry/Physics and Masters in Business Leadership
- Worked at Zisco Steel and Trojan Nickel Mine before joining Chemplex Corporation
- Rose to Group CEO of Chemplex in `February 2008`
- Former board directorships named on site include Zimbabwe Fertiliser Company, Sable Chemical Industries, Dyno-Nobel Zimbabwe, Business Council for Sustainable Development, and Zimbabwe Agricultural Society
- Recognised as Zimbabwe Institute of Management Private Sector Manager of the Year for `2008`
Engineer Thankful Musukutwa, Board Member
- Qualified engineer with MBA
- Public-service and mining/manufacturing leadership background named on site
- Institutions named on site include Office of the President and Cabinet, Ministry of Mines and Mining Development, Reserve Bank of Zimbabwe, Bindura Nickel Mine, The Wattle Company, Nyanga Pine, BHP Minerals Zimbabwe, and Zimbabwe Iron and Steel Company
- Past board membership named on site includes ART Corporation, TRANSLOAD Bio Diesel, Tuli Coal, and the Zimbabwe School of Mines
Dr Leonard Madzingaidzo, board-position profile page
- The board page repeats his CEO biography and also lists `ceo@sirdc.ac.zw`
Board members confirmed on the board page even where a full bio was not fully retrievable in the crawl:
- Professor Rudo I Makunike-Mutasa, Vice Chairperson
- Col Farai Taruvinga, Member
- Naume Mazango, Member
- Maureen Chitewe, Member
- Ottilia Murasi, Member
- Dr Nicholas Ndebele, Member
- Sakhile V Dube-Mwedzi, Member
- Mr Arthur D Maziveyi, Secretary
"""},{"text":"""# Built Environment and Transportation Systems Cluster

## Overview

The cluster refers to all the human-made surroundings where people live, work, and play, encompassing buildings, infrastructure, and public spaces (rural and urban). It involves the planning, design, construction, and management of these spaces and their relationship to human activities over time. Modern built environments are increasingly shaped by technological advancements and a focus on sustainability, accessibility, and well-being.
Figure 1 shows the programs under the cluster. Materials in the built environment and standards and by-laws and regulations are integrated in all the programs under the cluster.

## Products

- Fixed-dome bio gas digesters (different sizes)
- Rammed earth structure design and construction.

## Rural Development Infrastructure

Modern rural development focuses on improving the social, economic, and environmental well-being of people living in rural areas, often through a combination of infrastructure development, economic diversification, and sustainable resource management. This includes improving access to basic services such as education, healthcare, electricity, sanitation, and transportation (including feeder roads).
SIRDC is involved in the design and construction of rural infrastructure in the following areas:
- Sustainable Building Designs and Materials. Develop and promote sustainable building materials and designs that can withstand natural disasters and climate change, reducing the need for frequent repairs and replacements. This includes modern green houses, animal quarters, chalets, gazebos, shops, agro-based projects, lodges, houses and offices
- Rural Road Infrastructure. Improve rural road networks to enhance access to markets, healthcare, and education, thereby boosting economic growth and social well-being.
- Renewable Energy Systems. Implement renewable energy systems, such as solar and biogas, to provide reliable and clean energy to rural communities, reducing reliance on expensive and polluting diesel generators.
- Water and Sanitation Systems. Develop sustainable water and sanitation systems, including rainwater harvesting, greywater recycling and waste management, to improve rural livelihoods and reduce water-borne diseases.
- Sustainable Transportation Systems. Promote sustainable transportation systems to reduce dependence on fossil fuels and minimize environmental impact.
- Integrated Rural Planning. Foster integrated rural planning that balances economic, social, and environmental needs, ensuring that rural development is aligned with national goals and priorities.
- Integrated Water Management. Implement integrated water management systems that ensure equitable access to water resources for rural communities.
- Rural School Development. Support rural school development programs that provide safe, modern, and well-equipped learning facilities, enhancing access to quality education.

## Services

- Architectural and Structural drawings to include 3D visualization
- Building Information Modelling
- Geo-technical investigation
- Construction services
- Sustainable Facilities Management
- Facilities audits/ assessments
- Building Pathology
- SMART and Sustainable Master Planning for institutions, local authorities, farms, housing developments
- SMART Rural and Urban Planning
- Biogas Digester design, construction and maintenance
- Rammed earth construction
- Irrigation system design
- Rainwater and Greywater harvesting solutions
- Road Pavement Solutions
- Rural Infrastructure design and development.

## Sustainable Buildings and Cities

Sustainable buildings and cities have an environmental impact and promote social and economic well-being for current and future generations. This is achieved through eco-friendly practices, efficient resource management, and a focus on public health and quality of life.
The Cluster offers the following research and development and consultancy services:
SMART Master Planning
- Urban planning and development. Conducting research and consultancy on urban planning, transportation systems, and infrastructure development.
- Sustainable building design and construction. Research and Development on innovative and sustainable building materials, designs, and construction methods.
- Rammed Earth Construction. Design and construction of houses, schools and offices.
- Energy efficiency and renewable energy: Investigating ways to improve energy efficiency in buildings and promoting the use of renewable energy sources.
Asset Resilience & Life Cycle Management
- Building Pathology. Conducting building audits and assessments for existing buildings.
- Sustainable Facilities Management. Consultancy in the operational management and maintenance of buildings and their infrastructure.
Building Information Modelling
- Digital Twining. The creation and use of a connected, data-mirrored virtual copy of buildings and infrastructure to understand, analyze, and optimize their real-world performance.
- 3D Designs. Architectural and structural models for building and infrastructure.
- Model Making. The art and science of creating a physical or virtual, three-dimensional representation of an object, system, or space.
Geotechnical Soil Tests
This involves determining the physical, mechanical, and chemical properties of soil and rock formations at a project site.

## Training Services

Modern training in the built environment encompasses a wide range of topics focused on sustainability, technological advancements, and evolving industry practices. These training courses equip professionals with the skills to address contemporary challenges and contribute to a more efficient, resilient, and responsible built environment and transportation systems.
- Basic Construction Skills (Carpentry, Building and construction, Paving, Painting, Tiling)
- Sustainable Practices in Facilities Management
- Biogas Digester Construction and Maintenance.
- Rammed Earth Construction.
- Working at Heights and in Confined Spaces.
- Concrete Technology
- Project Management in Construction
- Irrigation Systems Design and Maintenance.

## Transportation Systems

Modern transportation systems are evolving rapidly, leveraging new technologies to enhance efficiency, safety, and sustainability. SIRDC is involved in research into smart transport infrastructure (roads, bridges, culverts, robots, etc..), transforming logistics and personal mobility. This is aimed at reducing traffic congestion, improve road safety, and minimize environmental impact through the integration of innovative technologies.
The Cluster focuses on the following areas:
- Transportation infrastructure development. Research and development to offer innovative solutions for transportation infrastructure and public transportation systems.
- Transportation systems optimization. Analyzing and optimizing transportation systems to improve efficiency, safety, and sustainability.
- Geotechnical Soil Tests. This involves determining the physical, mechanical, and chemical properties of soil and rock formations at a project site.
- Modern Road Network System Design. Research and Development in modern pavement materials and systems, such as pervious pavements, modified bitumen, prefabricated bridges and roads

## WASH and Waste Management

Modern waste management uses a comprehensive approach that utilizes advanced technologies and infrastructure to minimize environmental impact and promote a circular economy.
The cluster focuses on the following areas:
- Sustainable Waste Management Practices. Develop and promote sustainable waste management practices, including reduce, reuse, recycle (3Rs) strategies, to minimize waste generation and promote resource recovery.
- Waste-to-Wealth Initiatives. Explore waste-to-wealth initiatives, such as converting waste to energy, biofuels, or other valuable products, to create economic opportunities and reduce waste disposal costs. Design and construction of fixed dome biogas digesters and research into PVC-based portable biogas digesters.
- Greywater Treatment and Reuse. Develop innovative greywater treatment and reuse systems, enabling the safe reuse of treated greywater for non-potable purposes, such as irrigation or industrial processes.
- Solid Waste Management Infrastructure. Design and implement efficient solid waste management infrastructure, including waste collection, transportation, and disposal systems, to ensure proper waste handling and minimize environmental impacts.
- Circular Economy Approaches. Promote circular economy approaches that encourage the reuse and recycling of materials, reducing waste generation and promoting sustainable consumption patterns.
- Community-Based Waste Management. Implement community-based waste management initiatives that engage local communities in waste management activities, promoting behavioral change and community ownership.
- Rain Water Harvesting. Develop innovative rain water harvesting systems for housing, industrial rural and urban development.
"""},{"text":"""# SIRDC Cluster System and Strategy

## Unlocking Zimbabwe's Economy through coordinated Research and Development.

The Scientific and Industrial Research and Development Centre (SIRDC) has adopted a Cluster System to optimize delivery of services for the benefit of the economy in line with the “Whole of Government Approach”. The Research and Development (R&D) clusters, which fall under programme 2 mirror the National Development Strategy 1(NDS1) priority areas. These are: Agriculture; Energy and Power; Health; Industry and Manufacturing; Information and Communication Technologies; Built Environment and Transportation Systems; Water and Environment; Mining and Mineral Beneficiation; Small and Medium Enterprises; and Commercialisation.
The Programme 1 clusters which are Corporate Governance and Administration Clusters are Finance; Audit; Human Resources, Records Management, and Cafeteria; Works, General Services and Security; Administration. Procurement and Stores Management; Public Relations and Marketing; Hardware and Network Systems Administration; Logistics; and Legal Services.
This strategic re-alignment will enhance efficiency and effectiveness in the implementation of R&D programmes. The Centre will strongly leverage on Artificial Intelligence, Advanced Manufacturing Systems and Data Analytics capabilities to deliver ground breaking innovations in all Clusters, as we transition to the 5th Industrial Revolution and implementation of National Development Strategy 2.
"""},{"text":"""# Commercialisation Cluster

## Overview

The cluster is responsible for marketing and distribution of research and development outputs through Zimbabwe Technological Solutions (Pvt) Ltd, a wholly owned subsidiary of SIRDC.

## Branch Network

- Harare (SIRDC Premises, Hatcliffe)
- Bulawayo (ZITF Showground and Donnington West)
- Hwedza (Hwedza Town Centre)
- Beatrice (Beatrice Town Centre)

## Foundry Division

Established in 2005, the foundry business has steadily grown into a one-stop shop, providing a comprehensive range of products and services. Our operations cover every stage of the process from pattern work to molding, casting, machining and heat treatment.
Sectors serviced:
- Mining
- Infrastructural development
- Agriculture
- Automotive and Locomotive
- General Engineering
Product range:
- Grinding Media
- General Engineering Steels
- Grey Cast Irons
- Stainless Steels
- Manganese Steels

## Retail Division

The Division distributes and sells hardware and farm supplies including SIRDC Research Outputs:
- Sirdamaize Varieties (maize seed) and vegetable seeds
- Farm implements and accessories
- Building Materials
- Stock feed
- Animal health, livestock vaccines and antibiotics
- Agro Chemicals
- Foundry Products

## Seed Division

The Division produces and distributes climate-smart hybrid seeds that include:
- Maize seed varieties – Sirdamaize 113, 115 and 713.
- Diamond Potato Seed
- Rice
- Pearl and Finger Millet
- Sorghum
- Sunflower Seed
- Sugar Beans
Our Seed distributors:
- Farm and City
- Farmbiz
- Amtec Agro Centre
- Graniteside Hardware
- Farmers Hub
Processing PlantZTS installed a state-of-the-art seed processing plant that processes a variety of seeds including maize, soya bean and traditional grains such as sorghum and finger millet. Our high capacity is ideal for large scale production. The process includes seed cleaning, destoning, grading, coating and packaging.
"""},{"text":"""# Contacts, Access, and Service Standards

## Overview

Primary public contact details found on official pages and publications:
- Physical address: `1574 Alpes Road, Hatcliffe Extension, Harare`
- Postal address: `P O Box 6640, Harare, Zimbabwe`
- Switchboard: `0867 700 9663`
- Public Relations line: `0867 700 9670`
- Customer Care line: `0867 700 9906`
- WhatsApp: `+263 775 433 859`
- Business hours: `Monday to Friday, 08:00 to 16:30`
- Confirmed public email on charter and PR pages: `pr@sirdc.ac.zw`
Digital channels publicly listed in the client charter:
- Website: `www.sirdc.ac.zw`
- Facebook: `SIRDC`
- WhatsApp Channel: `SIRDC ZIMBABWE`
- X: `@sirdczim`
- LinkedIn: `SIRDC`
- Instagram: `@sirdczim`
- YouTube: `@sirdczim`
- TikTok: `@sirdc14`
- Pinterest: `@sirdcscientificindustrialresea`
Client service standards publicly stated in the 2025 Client Service Charter:
- Quotations or invoices: within `24 hours`
- In-person visits: attended to within `5 minutes`
- Telephone calls: answered within `3 rings`
- Verbal enquiries: within `5 minutes`
- Written enquiries: within `24 hours`
- Social media enquiries: within `3 minutes`
- Customer complaints: within `24 hours`
- Training target: at least `70%` satisfaction
Public complaint and feedback routes publicly stated in the charter:
- Contact the Public Relations Department
- Complete client satisfaction surveys
- Escalate complaints to Executive Management or the Board
- Use the complaints-lodging platform on the SIRDC website
"""},{"text":"""# Energy and Power Cluster

## Overview

The cluster focuses on research and development activities in the areas of energy management, automation and electrical, renewable energy and electronic systems.

## Automation and Electrical Systems

- Programmable Logic Controller (PLC) Programming
- Human Machine Interface (HMI) & SCADA Development
- Process Automation & Optimization
- Data Acquisition & Real-Time Monitoring
- Control Panel Design & Fabrication
- Power Generation, Transmission & Distribution system design.

## Electrical Systems Design and Installations (LV,MV,HV)

- Generator Installations and Maintenance
- Equipment and Machinery Maintenance and Repair Motor Selection, Installation & Maintenance
- Protection System Design (Circuit breakers, relays,fuses)
- Electrical Safety Assessments

## Electronics Systems

- Robotics & Motion Control Solutions (e.g.VSD/VFD, soft starters)
- Industrial loT Implementation
- System Integration (Sensors, actuators,drives, controllers)
- Circuit Design & Simulation (Analog and Digital) PCB Design & Fabrication
- Embedded Systems Development(Microcontrollers, FPGA programming)
- Sensor Integration
- Signal Processing & Communication Systems
- Consumer Electronics Development (Smart devices, loT products)
- Testing & Quality Assurance for electronic components.
- Failure Analysis & Product Optimization

## Energy Management

- Energy audits
- Power Quality Analysis including Power Factor
- Correction
- Energy bill Analysis
- Energy Management Systems Standard(ISO50001 EnMS) Implementation,
- Environment monitoring systems (dust, light,gases, noise, etc.).
- Lighting System Design

## Products

- Solar charge controller
- Electronic Door Lock
- Sine Wave Inverter
- Automatic Geyser Timer
- Laptop Charger
- Automatic Change Over Switch
- Battery Charger
- Microcontroller Programming Training Kit
- Bench instruments with digital displays (voltmeter, ammeter, mill ammeter, stop clock and digital thermometer)
- Signal generator
- Step variable power supply
- Electronic demonstration boards (logic gates, rectification, Schmitt trigger, operational amplifiers)
- LED displays (Rolling Message Displays, Score Boards, Sign Posts)
- Automatic Bell
- Biometric & RFID security system
- Egg Incubator
- Solar street light
- Solar system installation

## Renewable Energy Systems

- Solar projects Feasibility studies
- Designs and installations of backup systems
- Net metered systems design and installations
- Solar systems performance studies
- Solar systems projects management.

## Services

- Industrial process automation and control
- Boiler automation
- Volume measurement systems
- Industrial equipment re-engineering
- Embedded systems design and development
- Energy management
- Renewable energy systems
- Fuels
- Solar, electric and thermal energy Coal systems
- Energy audits and efficiency

## Training

- Programmable Logic Controllers (PLC)
- Variable Speed Drives (VSD)
- Embedded systems and microcontroller programming
- Energy management for industry
- Internet of Things /IoT
- Electronics circuit design
- Solar PV /Installation,Maintenance & Design
- PLC & HMI (Human Machine Interface)
"""},{"text":"""# ESG, Environmental, and Client-Trust Content

## Overview

The site exposes several environment/governance items not represented in the current local KB.

## ESG training feedback page

- The Environmental Science Institute reported an ESG training held from `17 to 20 June 2024`
- Target audience listed on the page included sustainability reporting officers, CSR officers, SHEQ representatives, engineers, sustainability consultants, managers, private companies, SMEs, trade associations, and interested individuals
- The same page announced a follow-on ESG training window from `29 July to 1 August 2024`
- Inquiry contacts named on that page:
- Lilian Wisikoti: `+263 773 080 687`, `lwisikoti@sirdc.ac.zw`
- Netsayi Ngorima: `+263 775 215 995`, `nngorima@sirdc.ac.zw`
- Dr Farai Matawa / ESI contact block also listed

## Environmental Science Institute page additions

- ESI hosts the National Cleaner Production Centre of Zimbabwe
- ESI offers environmental consultancy, assessments, cleaner production work, industrial hygiene, wastewater/water treatment, waste management, ESG, ESIA, air quality, and remediation work
- The page lists project examples including work with UDCORP, IDBZ, World Vision, ZCDC, Africure Pharmaceuticals, Invictus Energy/Geo Associates, Lafarge, Delta Beverages, and others
- ESI collaboration interests listed include wastewater facilities design, climate-change implementation, contaminated-water/soil remediation, industrial hygiene, flue-gas modelling, and laboratory establishment

## Ethics and reporting channel discovered in the official news archive

- SIRDC publicly announced subscription to `Axcentium EthicsLine`
- Audience explicitly includes clients, suppliers, employees, members of the public, and other stakeholders
- Contact routes named on the page:
- Econet toll free: `0808 5500 / 4461`
- NetOne toll free: `0716 800 189 / 0716 800 190`
- Telecel toll free: `0732 220 220 / 0732 330 330`
- WhatsApp: `0772 161 630`, `0718 267 886`
- Email: `reports@axcentiumethicsline.co.zw`
- Website: `www.axcentium.co.zw`

## What ESG means, according to SIRDC

- ESG is explained on the site as Environmental, Social, and Governance criteria used to assess environmental impact handling, stakeholder relations, and governance quality
- The page argues ESG is increasingly relevant for investor decision-making, access to capital, long-term resilience, and compliance
"""},{"text":"""# Health Cluster

## Overview

The cluster focuses on the One Health approach, which recognizes the deep interconnection between human, animal, and environmental health, promoting integrated efforts across sectors to prevent and manage health risks, including emerging zoonotic diseases.

## Disease Control & Pandemic Preparedness

- Management of disease outbreaks and health emergencies.
- Continuous surveillance of disease patterns.
- Rapid response teams for immediate outbreak control.
- Public awareness campaigns on symptoms and prevention.
- Collaboration with community stakeholders for effective response.

## Food Safety & Nutrition

- Monitoring of potable water and environmental quality.
- Regular quality checks on food raw materials and products.
- Development of authenticity tests to detect food adulterants.
- Training programs for meat processing and food safety.
- Innovation in food products, like fortified cereals and herbal blends.

## Infectious Diseases Management

- Focus on priority diseases: HIV, TB, malaria, cholera, and more.
- Development of rapid diagnostic tests for malaria species.
- Strengthening local health responses and disease management.
- Community education on prevention and treatment of infectious diseases.
- Research on improving diagnostic capabilities for major infections.

## Laboratories & Testing Unit

- Development of rapid diagnostic tests for various pathogens.
- Creation of biosensors for pathogen detection, including AMR.
- Implementation of Point-of-Care microbial testing.
- Enhancing access to microbiological testing facilities.
- Continuous improvement of laboratory testing capabilities.
- Data-sharing platforms for labs to strengthen research collaborations.

## Non-Communicable Diseases (NCDs)

- Focus on prevention, early detection, and management of NCDs.
- Research on lifestyle factors contributing to NCD prevalence.
- Identification of biomarkers for major cancers (e.g., cervical, breast).
- Development of screening programs for early detection.
- Advocacy for establishing a Cancer Research Laboratory in Zimbabwe.

## Occupational Health & Safety

- Development of advanced personal protective equipment (PPE).
- Environmental monitoring through Internet of things (IoT) to ensure workplace safety.
- Implementation of fatigue and impact detection systems in helmets.
- Training programs on safety protocols and hazard recognition.
- Regular assessments of workplace conditions to reduce risks.
- Industrial effluent testing.

## Primary & Secondary Healthcare

- Addressing chronic drug shortages and ensuring medication access.
- Validation of traditional knowledge and complementary medicines.
- Promoting the consumption of medicinal herbs through product development.
- Public awareness campaigns on the benefits of herbal remedies.
- Collaborating with local pharmacies and healthcare providers for better service.

## Products

- Traditional Grains products (Zviyo Porridge, Zviyo (Finger Millet) Meal, Mutakura Mix, Hupfu, Mhunga, Hupfu-Zviyo & Hupfu-Mapfunde)

## Services

- Food technology
- Biomedical technology
- Industrial biotechnology
- Biochemical and chemical engineering
- Analytical services (pharmaceuticals, veterinary drugs, pesticide residue, soil, water, food, feed, geo-chemistry)
- Indigenous food processing
- Herbal oils production and detergent production
- Quality control of raw materials and products
- Contract research and development

## Training Services

- Good Laboratory Practices
- Gas Chromatography (GC-FID)
- High Performance Liquid Chromatography (HPLC-DAD)
- Flame Atomic Absorption Spectroscopy (FAAS)
- Meat Processing: Biltong and Sausage Making
- Food Safety & Food Defence Along the Food Chain
- Vegetables Preservation and Pickling
- Sequencing and Sequence Data analytics
- Industrial Biosafety

## Vaccine Unit

- Localization of livestock vaccine production to enhance availability.
- Focus on preventing diseases: Theileriosis, Newcastle disease, Blackleg, Lumpy skin, Anthrax, Foot and Mouth Disease.
- Research on vaccine efficacy and safety.
- Community outreach programs to promote vaccination awareness.
- Collaboration with veterinary services for vaccination campaigns.

## eHealth Innovations

- Utilization of digital technologies to enhance healthcare delivery.
- Implementation of electronic health records (EHR) for better patient management.
- Development of mobile health applications for chronic disease management.
- Promoting telemedicine for patients in underserved areas.
- Enhancing patient engagement through educational resources.
- Address cybersecurity and data protection in electronic health systems.
- Explore integration with wearable health monitoring devices.
"""},{"text":"""# Industry and Manufacturing Cluster

## Overview

Focuses on developing innovative technologies and processes to enhance industrial competitiveness and manufacturing capabilities. It includes calibration of industrial equipment, 3D designing, production and maintenance of industrial equipment.

## Agricultural Engineering Automation

- Design and implementation of automated irrigation systems
- Development of precision agriculture technologies (e.g., GPS-guided tractors, drones)
- Automation of harvesting, planting, and cropmonitoring equipment.
- Integration of IoT sensors for soil, weather, and crop health monitoring
- Smart greenhouse and livestock system automation (e.g., climate control, feeding)

## Automation and Robotics

- Design and programming of robotic arms for assembly and packaging
- Integration of vision systems for quality control and inspection
- Development of autonomous mobile robots for material handling
- Use of collaborative robots (cobots) for safe human-machine interaction.
- Troubleshooting & upgrading of robotic systems for higher efficiency.

## Biomedical Technologies & Engineering

- Design and development of medical devices(e.g., ventilators, diagnostic tools)
- Development of prosthetics and orthotics using advanced materials and 3D printing
- Automation of diagnostic and therapeutic equipment
- Integration of health monitoring systems andwearable devices
- Design of assistive technology for rehabilitationand mobility

## Calibration

- Measurement and adjustment of instrumentsand sensors to ensure accuracy
- Calibration of pressure, temperature, flow, andelectrical instruments
- Documentation and traceability according to ISO standards
- Use of reference standards and certified equipment
- Periodic review and verification of calibrationintervals and records.
Research and Development Projects
- Biomedical Engineering
SIRDC uses engineering, biology, and medicine knowledge to design and analyze solutions to problems in healthcare such as:
- Prosthetics (hands, legs, etc);
- Neonatal Incubator;
- Telemedicine, Continuous Glucose Monitoring and Control Device;
- Motorized Wheelchair.
- Digital Manufacturing
SIRDC researches and develop prototypes and products for various industries, through digital manufacturing (3D Printing) in tandem with the Industry 4.0 emerging technologies.
- Agricultural Engineering (Automation, AI)
SIRDC develops and integrate innovative, smart, and sustainable technologies in agriculture.
- Material Handling Robotic Arm
SIRDC is developing automated systems and robotic solutions to enhance operational efficiency, improve product quality, ensure safety and drive innovation.
Industrial Support Services
- Calibration of Measuring Equipment
SIRDC through the National Metrology Institute provides traceable calibration services to private as well as public sectors of the economy. The services enable these companies to compete on the global markets as well as meet statutory and contractual obligations.
- Quality Management Systems
SIRDC through the National Metrology Institute develops and Maintain quality management systems based on ISO/IEC 17025.

## Concurrent Engineering

- Cross-functional team collaboration from concept to production
- Parallel development of product design andmanufacturing processes
- Design for Manufacturability (DFM) and Designfor Assembly (DFA) practices
- Early inclusion of customer feedback and qualityassurance input
- Use of digital tools for real-time design collaboration and simulation.

## Equipment Service, Repair and Maintenance

- Routine preventive and predictive maintenanceof machinery
- Troubleshooting and repair of electrical, hydraulic, and mechanical systems
- Documentation and tracking of service history
- Spare parts management and failure analysis
- Upgrade or retrofitting of obsolete equipment forbetter performance

## Installation & Commissioning

- Assembly and setup of machinery and industrialequipment on-site
- System integration and calibration during startup
- Functional testing to ensure compliance with design specifications
- Training of operators and maintenance personnel during handover
- Documentation of commissioning proceduresand performance validation.

## Lean Manufacturing

- Implementation of 5S workplace organizationtechniques
- Value Stream Mapping to identify and eliminate waste
- Continuous improvement (Kaizen) initiatives
- Just-In-Time (JIT) production planning
- Standardization of processes to reduce variation and increase efficiency

## Products

- Gravity grinding mill (electric, diesel, hybrid)
- Dehuller
- Chicken plucker
- Grain separator & Thresher

## Services

- Equipment calibration services
- Research and development of measurement standards
- Installation and commissioning of new equipment
- Development of quality management systems, e.g. ISO/IEC 17025, ISO 9000
- Plant and equipment design and development
- Plant construction and commissioning
- Lean management
- Digital Manufacturing (3D Printing, CAD 3D Modelling & Slicing)
- Product design technologies
- Manufacturing process optimisation
- Engineering maintenance and reliability
- Design and development of machine tools Packaging, design and production
- New materials: polymers

## Training

- Metrology
- Reliability Centred Maintenance
- Lean Six Sigma
- Computer Aided Design
- Pump Selection and Design
- Conveyor Selection and Sizing.

### Plant / Process Design and Automation

- Designing efficient production layouts and material flow systems
- Development of automated control systems(PLCs, SCADA) for process optimization
- Integration of safety and compliance systems indesign. Simulation and modeling of manufacturing processes before implementation.
- Automation of repetitive or hazardous tasks toimprove productivity and safety.
"""},{"text":"""# Information and Communication Technology Cluster

## Overview

The cluster drives innovation and consultancy in software development, networking, cybersecurity, cloud computing and data systems to enable breakthroughs in Artificial Intelligence, quantum computing, and next generation networks that transform industries, empower societies, and address global challenges.

## Artificial Intelligence and Robotics

- Chatbots, Natural Language Processing and Computer vision
- Machine Learning: (Predictive Analytics, Reinforcement Learning, Sentiment Analysis)
- Automation (Robotic Assistants, Self-Navigating Robots, Human-Robot Interaction, Automated Manufacturing, Assembly Lines)
- AI Hardware & Embedded Systems (AI chips, Firmware, Printed Circuit Board (PCB) design)
- Autonomous Drones
- AI-powered Robotic Assistants (Healthcare, Logistics)

## Big Data

- Data Analytics & Visualization Dashboards
- Data Mining
- Data Warehouse
- Real-Time Data Processing Pipelines
- Business Intelligence
- Large-Scale Data Storage & Management

## Cybersecurity & Cryptography

- Threat Detection & Prevention
- Anomaly Detection in Network Security
- Penetration Testing
- Encryption

## ICT Infrastructure

- Networking (Network Design, Configuration, Installation, Maintenance, Audit and Upgrade)
- Repair and Maintenance

## Products

- Computerised Maintenance Management System (CMMS)
- Computerised Asset Management System (CAMS)
- Insurance Broking System (IBS)

## Services

- Design and development of high quality bespoke software systems
- Website design and development
- Network Design & Support
- ICT Repair & Maintenance

## Smart Automation & IoT Integration

- Smart Home/Office Automation Systems
- AI-Based IoT Sensors & Monitoring
- Predictive Maintenance for Industrial Equipment
- Connected Vehicle & Fleet Management Solutions

## Software Engineering

- Mobile Application Development
- Desktop application & Web Development

## Training & Support

- Artificial Intelligence (AI)
- Data Analytics
- Software Development
- Database & Data Management
- Cyber Security
- Digital Marketing
- Networking
- Hardware
"""},{"text":"""# Management and Leadership Directory

## Directors and senior managers

Tarisayi Zvoma, Director: Public Relations & Marketing
- Joined the wider SIRDC stable in `2007`
- Previously served in Parliament of Zimbabwe public education and public relations roles
- Site attributes experience in corporate communications, advocacy, media liaison, and event management
- Public email: `tzvoma@sirdc.ac.zw`
Dr Philemon K Kwaramba, Director: Partnerships & Resource Mobilization Unit
- Site says he has served SIRDC for more than `26 years`
- Former Director of Business Operations Unit (`2005-2023`)
- Background spans agricultural economics, development economics, NHIR/Blair Research Laboratory, ICRISAT Southern Africa, and Friedrich Ebert Foundation
- Represents SIRDC on international networks including industrial and technological research organization networks and science/technology park networks
- Public email: `pkwaramba@sirdc.ac.zw`
Ambrose Kavu, Director: TIPS
- Public email listed: `akavu@tips.org.zw`
Engineer Fred Gweme, ICT/technical-support leadership
- The site says ICT was separated from the Informatics Institute in `February 2024`
- Eng Gweme became Director of ICT and then Director, Technical Support and Quality Assurance in `February 2026`
- Background named on site: computer engineering, electronics/comms product design, commercialisation of teaching equipment, national e-waste task-team work
- Public email: `fgweme@sirdc.ac.zw`
John Nyamukondiwa, Manager: Audit
- Audit Manager since `July 2023`
- More than `17 years` audit experience according to the site
- Public email: `jnyamukondiwa@sirdc.ac.zw`
Nomagugu Madlunga, Head: Legal Services
- Public email: `nmadlunga@sirdc.ac.zw`
Barbra Nyakurerwa, Manager: Cafeteria
- Public email: `bnyakurerwa@sirdc.ac.zw`
Tendai Chiroodza, Manager: Central Mechanical Department
- Public email: `tchiroodza@sirdc.ac.zw`
Mufudzi S Chigowe, Manager: General Services
- Public email: `mchigowe@sirdc.ac.zw`
Sifiso Mucheki, Manager: Records
- Name and role are visible on the management page

## Executive leadership

Dr Leonard Madzingaidzo, Chief Executive Officer
- CEO since `1 July 2023`
- Joined SIRDC in `February 2000` as a Research Scientist in the Biotechnology Research Institute
- Later led the Food and Biomedical Technology Institute and became Executive Director Technical in `January 2005`
- Prior work included Dairy Marketing Board, Standards Association of Zimbabwe, and Cairns Foods
- Experience areas named on the site: manufacturing, quality assurance, research and development, technology transfer, commercialisation, and intellectual property management
- Academic background named on the site: BTech (Hons) Applied Biology and Biochemistry, MSc Biotechnology, and PhD Biotechnology/Biochemical Technology from BOKU Austria
- Current affiliations named on the site: board member of Standards Association of Zimbabwe and advisory board member of KOPIA
- Public email listed on site: `lmadzingaidzo@sirdc.ac.zw`
Prof Tawanda Mushiri, Executive Director (Technical)
- In post from `1 March 2024`
- Academic background named on the site: Mechanical Engineering, Manufacturing Systems and Operations Management, and PhD in Engineering with focus on fuzzy-logic-based AI systems
- Site says he holds certifications in AI, Data Science, Machine Learning, and Deep Learning
- Research areas named on site: robotics, AI, biomedical engineering, robotic first-aid systems, disease-prediction modelling using AI
- Site says he has managed projects and grants up to `USD 3.5 million`
- Public email listed on site: `tmushiri@sirdc.ac.zw`
Gladys G Mudyahoto, Executive Director (Finance & Administration)
- Elevated to this post in `December 2017`
- Previously Financial Controller and Board Secretary since `2003`
- Also Principal Officer of the SIRDC Pension Fund
- Competencies named on site: financial planning and management, project management, strategy implementation, pension, payroll, and tax administration
- Public email listed on site: `gmudyahoto@sirdc.ac.zw`

## ZTS leadership

Selwyn D Dhliwayo, General Manager
- Site says he studied Economics, Finance, and Portfolio Management at the University of Cape Town
- Background across public sector, insurance, and financial services
- Public email: `sdhliwayo@sirdc.ac.zw`
Austina Shereni, Head: Retail Division
- Site says she worked through sales/marketing roles before becoming acting retail manager
- Background includes MBA, BCom Marketing, and digital-marketing/commercialisation training
- Public email: `ashereni@sirdc.ac.zw`
Kudzanayi Tsokodayi, Manager: Seed Division
- Background named on site: technical sales, agricultural consulting, seed production agronomy, former farming operations management
- Public email: `ktsokodayi@sirdc.ac.zw`
Gilbert Mucheri, Manager: Business Development
- Site says he has over `23 years` of experience
- Background spans ZESA and multiple sales/marketing/operations roles
- Public email: `gmucheri@sirdc.ac.zw`
Peter Chimburi, Acting Manager: Foundry
- Site presents foundry training/certification background and operational experience in furnace work, moulding, core-making, fettling, and scrapyard management
"""},{"text":"""# Mining and Mineral Beneficiation Cluster

## Overview

The cluster carries out research and development in mining, mineral processing technologies, polymer and material sciences, embracing circular economy.

## Circular Economy & Waste Valorisation

Turning Industrial Waste, such as tailings, slag, red mud, fly ash, and waste rock, into materials for green infrastructure.
- Converting waste into aggregates, geopolymer cements, tiles, and bricks, reducing reliance on carbon-intensive and virgin materials
- Reduced CO₂ emissions via cement substitution
- Job creation in recycling & local manufacturing
- Supporting a global shift to waste-to-value innovation

## Low-Carbon Metallurgy: The Future of Steel

Decarbonizing Zimbabwe's Steel Sector
- Use of local low-grade iron ores
- Self-reducing pellet, composite pellets decreases usage of coke & emissions
- Cleaner smelting processes for sustainable infrastructure
- Access to premium global green steel market

## Material Characterisation & Geochemical Analysis

Enabling Smart Exploration & Industrial Quality Assurance
- Advanced testing of ores, soils, and materials (metals, polymers, etc.)
- Services for resource estimation, environmental monitoring & process optimisation.
- Supporting mining, industry, manufacturing, energy, and construction sectors
- Traceability, durability, and global compliance.

## Mineral Beneficiation

Positioning Zimbabwe in the Global Clean Energy Value Chain
- Advanced processing of critical minerals, PGMs, precious metals & diamonds.
- Applications in hydrogen fuel cells & catalytic converters.
- Local refining expertise for high-value, export-ready products.
- Scalable, research-driven solutions aligned with sustainability goals.

## Mineral Geo-Mapping & Geoinformation Systems

Building a Data-Driven Mineral Economy
- High-resolution geological mapping and mineral targeting
- Integration of GIS and remote sensing for real-time exploration insights
- Supporting national land use planning, investment decisions, and sustainable mining practices
- Enhancing transparency, risk management, and resource governance.

## Products

- Ball mill components ( mill balls, liners, pinion & girth gears)
- Jaw crusher lining
- Stamp mill components ( shoes, cam, tappets, heads, dies)
- Cast products in steel and iron
- Agricultural, construction and automotive products

## Services

- Mine feasibility studies
- Metallurgical plant optimisation
- Physical metallurgy and foundry technology
- Extractive metallurgy and mineral processing

## Training

- Materials Characterization Techniques – FTIR, SEM, XRD, DSC, TGA for analysis of polymers and minerals.
- Corrosion Testing & Prevention – Focus on mining and agro-industrial environments
- Metallurgical Failure Analysis – Applied techniques to investigate product failures in mining & engineering systems.
- Mechanical Properties Testing – Hardness, tensile, impact, and wear behaviour for foundry and composite materials.
- Electronic Waste Management
- Plastic Waste Upcycling
- Green Mineral Beneficiation – Use of non-toxic reagents and renewable processes
- Bio-based Polymers and Additives – Sourcing from natural materials (e.g., cassava, starch, plant fibers)
- Sustainable Foundry Practices – Sand reuse, fume capture, and energy optimization
- Techno-economic Feasibility for Beneficiation Projects
- Small-scale Manufacturing Start-up – Polymer products, mineral processing, foundry casting
- Business Models for Material-based SMEs – Costing, marketing, regulatory compliance
- Mineral Processing Techniques – Crushing, screening, grinding, and flotation
- Advanced Ore Characterization – Mineralogical and chemical analysis for beneficiation planning
- Tailings Reprocessing & Remediation – Safe reuse and recovery of valuable minerals from mine waste
- Small-Scale Mining Operations & Safety – Regulatory compliance, beneficiation setups, environmental safeguards
- Mine Water Management & Acid Drainage Control
- Polymer Processing & Compounding – Extrusion, injection molding, calendaring, and mixing
- Adhesives, Coatings & Paints Technology – Formulation chemistry and surface application methods
"""},{"text":"""# Official News and Event Additions

## Overview

These items are new website-derived facts that are not already well captured in the current local text set.

## 2026 to 2030 Strategic Planning Workshop

- The official site says the workshop was held in `Bulawayo` at `Cresta Churchill`
- Participants named on the page: Board, Executive, Directors, Managers, and Staff
- The workshop was facilitated by `PSC Academy`
- The purpose stated on the page was to review the `2021-2025` Strategic Plan and prepare the `2026-2030` plan in line with `National Development Strategy 2`
- The official page explicitly says Board Chairperson Misheck S Kachere stressed the need to employ higher skills and technology and to move SIRDC from production-push to market-centred thinking

## International and partnership-facing news found through the official archive

- `SIRIM Berhad welcomes delegation from SIRDC`: official site says the visit focused on stronger ties and knowledge exchange
- `SIRDC meets China National United Equipment Group (CNUE)`: official site says the meeting focused on enhancing cooperation around compact equipment
- `Zimbabwe-Tanzania benchmarking exercise`: the site says a Tanzanian Ministry of Science and Technology team visited SIRDC, including tours of vaccine laboratories and metrology laboratories

## Research, survey, and outreach items found in the official archive

- `Clients Satisfaction Survey`: SIRDC announced an independent survey through Winfield Strategy and Innovation
- `The 13th Zimbabwe International Research Symposium`: site says SIRDC participated, Prof Tawanda Mushiri presented on coal beneficiation, and the event theme was sustainable industrialisation

## ZAS 2025 related coverage

- The archive/search results show that the direct `sirdc-executives-tour-zas-2025-stands` URL was unstable, but the official site clearly indexed a related entry titled `SIRDC executives tour #ZAS 2025 stands`
- The archive also indexed `ZAS 2025 Day 04`
- The `ZAS 2025 Day 04` page showed active participation from ZTS, Kopia staff, agriculture staff, built environment staff, ICT staff, water/environment staff, and executives visiting the stand

### ZAS 2024 / business-conference coverage

- `SIRDC Wins Bronze at ZAS 2024`: the site says SIRDC placed third in the State Enterprises category
- The same article states products showcased included foundry castings, packaged herbal teas and potted plants, maize seed, value-added food-crop products, and fabricated machinery
- The article also highlights the mobile metrology laboratory and ESG-training synopsis sessions at the stand
- `SIRDC Participates at the ZANU PF Business Conference`: the site says SIRDC, TIPS, and ZTS exhibited together, and that high interest centered on mushroom training and Sirdamaize
"""},{"text":"""# Public Relations Department

## Overview

- Public Relations is positioned as a management function for:
- building corporate image
- establishing mutually beneficial relationships with stakeholders
- safeguarding corporate identity
- maintaining media links
- producing company profiles, annual reports, and public literature
- organising corporate events
- showcasing and promoting the Centre's activities
- maintaining information channels including the website
- developing and implementing the social media strategy
Library and information services under PR:
- reference services and internet browsing
- access to research journals in print and online
- access to current daily and weekly local newspapers
- pictorial reference sources
Confirmed PR contact details:
- Director contact line: `+263 772 423 719`
- Emails: `pr@sirdc.ac.zw`, `tzvoma@sirdc.ac.zw`
"""},{"text":"""# Publications and Download Inventory

## Overview

Indexed publication/download titles found:
- `Partitioned Operating Spaces for MSMES in Zimbabwe`
- `2025 SIRDC Client Service Charter`
- `SIRDC 2025 ZITF E-Book`
- `SIRDC 2025 Solusi E-Book`
- `2024 Annual Report`
- `Africonfex Expo 2024`
- `2024 Mine ENTRA E-Book`
- `2024 ZAS E-Book`
- `2023 Annual Report`
- `State of Manufacturing Micro, Small and Medium Enterprises in Zimbabwe 2023`
- `Zimbabwe 2023 Career Guidance Expo`
- `2022 Annual Report`
- `State of Manufacturing Micro, Small and Medium Enterprises in Zimbabwe 2021`
"""},{"text":"""# Small and Medium Enterprise Cluster

## Overview

The cluster supports SMEs through research, assessments, capacity building, and technology and innovation support, while also promoting formalisation and influencing policy to create an enabling environment for sustainable growth.

## Commercialisation of Technologies

- Market Access Facilitation
- Networking Opportunities
- Funding and Investment Resource
- Enhanced Resource Utilisation

## SME Information Hub & SME Databases

- Centralized Information Repository
- Database
- Performance Benchmark
- Technology and Innovation Support
- Policy Advocacy

## SME Research

- National Research and Assessment
- Mini Surveys
- Policy Analysis
- Research Consultancy

## SMEs Capacity Building

- Business Skills Training
- Technical Skills Training
"""},{"text":"""# Current Training Catalogue and Registration Page

## Agriculture

- Herbs training course: `$40`
- Horticulture training course: `$40`
- Livestock Production training course: `$40`
- Mushroom Production training course: `$40`
- Tissue Culture training course: `$60`

## Built Environment

- Artificial Intelligence in the Built Environment: `$250`
- Climate Change Management: `$250`
- Irrigation Systems: `$250`
- Sustainable Facilities Management: `$300`

## Energy and Power

- Advanced PLC & HMI Programming: `$300`
- Advanced Solar Photovoltaic Systems Design, Installation, Commissioning: `$150`
- Basic Electronics Circuit Design: `$150`
- Basic Micro-Controller Programming: `$150`
- Basic Printed Circuit Board (PCB) Fabrication: `$150`
- Basic PLC Configuration and Programming: `$150`
- Basic Variable Speed Drive (VSD) Configuration: `$50`
- Energy Management for Industries: `$150`
- Introduction to AOT: `$150`
- Robotics: `$150`
- Supervisory Control and Data Acquisition (SCADA): `$150`

## Health

- Flame Atomic Absorption Spectroscopy (FAAS): `$200`
- Food Safety and Food Defence Along the Food Chain: `$150`
- Gas Chromatography (GC-FID): `$250`
- High Performance Liquid Chromatography (HPLC-DAD): `$250`
- Meat Processing: Biltong and Sausage Making: `$100`
- Sequencing and Sequence Data Analytics: `$800`
- Vegetables Preservation and Pickling: `$150`

## Industry and Manufacturing

- Advanced Planned Maintenance (software based): `$300`
- Advanced Production and Operations Management (software based): `$300`
- Advanced Reliability Centered Maintenance (software based): `$300`
- Bearing Technologies: `$300`
- Cleaner Production Technologies: `$300`
- Computer Aided Design (Drawing): `$300`
- Condition Based Monitoring: `$300`
- Gas Welding: `$500`
- Hydraulic and Pneumatic Systems: `$300`
- Lean Six Sigma: `$300`
- Planned Maintenance Management: `$300`
- Production & Operations Management: `$300`
- Pump Selection and Sizing: `$300`
- Reliability Centred Maintenance (RCM): `$300`
- Reliability Engineering and Centred Maintenance: `$300`
- Total Productive Maintenance (TPM): `$300`

## Information Technology

- Big Data and Internet of Things: `$200`
- Cybersecurity: `$200`
- General Data Analytics: `$200`
- Generative AI for Managers: `$200`
- Machine Learning: `$200`
- Mobile Application Development: `$200`
- Software Development: `$200`

## Mining and Mineral Beneficiation

- Basics of Screw Extrusion: `$250`
- Electronic Waste Management: `$150`
- Fundamentals of Foundry Technologies: `$150`
- Introduction to Gold Mining: `$180`
- Lithium Benefication: `$250`
- Paint and Coatings Applications: `$180`
- Plastic Recycling Fundamentals: `$150`
- Safe Mining Practices for Small-Scale Miners: `$200`
- Small Scale Mining As A Business Venture: `$200`

## Small and Medium Enterprise

- E-views: `$100`
- Horticulture Export Development: `$100`
- Mushroom Production: `$40`
- Oyster Mushroom Production: `$50`
- Retirement Planning: Agribusiness Income Generating Activities: `$1,000` per organisation
- Statistical Package for the Social Sciences (SPSS): `$100`
- WordPress Website Development: `$100`

## Water and Environment

- Artificial Intelligence in Environmental Systems Management: `$180`
- Climate Change Management: `$250`
- Environmental and Social Impact Assessment (ESIA): `$300`
- Environmental, Social and Governance (ESG): `$300`
- Geographic Information Systems (GIS): `$350`
- Green House Gases (GHG) Inventory: `$180`
- Resource Efficiency and Cleaner Production (RECP): `$180`
- Waste & Effluent Management (WEM): `$180`
"""},{"text":"""# Water and Environment Cluster

## Overview

The cluster provides environmental and geospatial research, consultancy, and training, supporting sustainable industrialisation, community empowerment, and policy decisions through advanced geospatial technologies and data-driven solutions.

## Circular Economy & Waste Solutions

- Recycling and resource recovery initiatives,
- Development of eco-innovative solutions and products for industry.

## Climate Change Solutions

- GHG inventories, emissions reduction plans,
- Renewable energy integration (solar, biogas),
- Wetland restoration and biodiversity conservation.

## Consultancy Services

We provide tailored solutions in:
- Comprehensive assessments for mining, agriculture, energy, and infrastructure projects, and
- Mitigation of environmental and social risks to ensure compliance and sustainability.

## Environmental Analytics & Monitoring

- Air, water, and soil quality testing and analysis,
- Emissions modelling and wastewater solutions,
- Cleaner production strategies to reduce pollution

## Geo-Information & Remote Sensing Solutions

- Satellite imagery and GIS-based analysis,
- Spatial modelling, WebGIS, and real-time monitoring,
- Agro-environmental mapping, risk modelling, and geodatabases development.

## Products

- Geo-Database including cadastres

## Research & Development

We lead innovative research in:
- Climate change adaptation strategies and circular economy models,
- Environmental impact and biodiversity assessments,
- Wetland restoration, mine rehabilitation, and pollution control,
- Advanced remote sensing and GIS for monitoring of the Environment,
- Spatial data analysis and
- Development of operational WebGIS systems for geospatial solutions

## Restoration and Bioremediation

- Mine rehabilitation and closure planning, and
- Soil remediation for degraded sites

## Services

- Environmental impact assessments
- Mine closure and rehabilitation
- Biodiversity studies
- Industrial hygiene and environmental monitoring
- Solid waste and wastewater management
- Monitoring and evaluation
- Baseline studies
- Resource efficient and cleaner production audits
- Climate change mitigation and adaptation.
- Agriculture and forestry applications
- Environmental systems analysis
- Creation and management of GIS (spatial) databases
- Geo-referencing and digitising data
- Mineral exploration using GIS, geo-statistics and remote sensing techniques
- Water resources inventories, management and hydrological modelling
- Early warning of natural disasters Satellite imagery processing and classification for land-use /land-cover change monitoring
- Digital near-real time data collection
- Creation of digital data collection aggregate servers (internet or LAN based) Creating land management information systems, e.g. Cadaster
- GPS field-area calculations and GIS mapping
- Project monitoring and evaluation using GIS and remote sensing techniques.

## Training & Capacity Building

We deliver practical, hands-on training in:
- Climate and sustainability reporting,
- Greenhouse gas (GHG) accounting and mitigation,
- Waste, water, and effluent management
- Geospatial applications of Remote sensing & spatial analysis, and
- ESG integration and sustainable industrial practices.
- Environmental Management System (EMS)
- Climate Finance
Our courses are designed for professionals in industry, government, NGOs, and academia.
"""},{"text":"""# Commercialisation, ZTS, and Product-Support Notes

## Overview

The supplied ZTS-related URLs were surprisingly sparse in direct crawl, but the official site still exposes useful structured facts.
New or under-documented website facts:
- ZTS is clearly positioned throughout the site as the commercial entity for SIRDC outputs
- The management page exposes current ZTS leadership roles:
- General Manager
- Accountant
- Head: Retail Division
- Manager: Business Development
- Manager: Seed Division
- Acting Manager: Foundry
- The current publications and management pages reinforce that SIRDC is actively packaging its products/services into e-books, promotional materials, and commercial registration pages
Important limitation:
- the direct ZTS landing pages discovered through indexing were sparse during this crawl and did not expose much descriptive text beyond headings and navigation structure.
"""},
]

# ============================================
# FIX: Helper functions for greeting detection
# ============================================

# Greeting responses cache
GREETING_RESPONSES = {
    "hi": "Hello! I'm RALPH, your SIRDC assistant. How can I help you today?",
    "hello": "Hello! I'm RALPH, your SIRDC assistant. How can I help you today?",
    "hey": "Hey there! I'm RALPH. What can I help you with about SIRDC?",
    "how are you": "I'm doing great, thanks for asking! How can I assist you today?",
    "how are you doing": "I'm doing great, thanks for asking! How can I assist you today?",
    "what's up": "Not much, just here to help! What can I do for you?",
    "whats up": "Not much, just here to help! What can I do for you?",
    "good morning": "Good morning! How can I assist you today?",
    "good afternoon": "Good afternoon! How can I help you with SIRDC?",
    "good evening": "Good evening! What can I help you with?",
    "thanks": "You're welcome! Is there anything else I can help you with?",
    "thank you": "You're welcome! Let me know if you need anything else.",
    "bye": "Goodbye! Feel free to come back if you have more questions about SIRDC.",
    "goodbye": "Goodbye! Have a great day!",
    "ping": "Pong! I'm here and ready to help.",
}

SIMPLE_RESPONSES = {
    "whats your name": "I'm RALPH, your SIRDC assistant.",
    "what is your name": "I'm RALPH, your SIRDC assistant.",
    "who are you": "I'm RALPH, your SIRDC assistant for SIRDC questions.",
    "what can you do": "I can answer questions about SIRDC, its clusters, services, products, training, and related topics.",
    "help": "I can answer questions about SIRDC, its clusters, services, products, training, and related topics.",
    "thanks": "You're welcome!",
    "thank you": "You're welcome!",
}

def normalize_query(query: str) -> str:
    """Normalize user text for simple cache lookups."""
    lowered = query.lower().strip()
    lowered = lowered.replace("'", "")
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered

def is_greeting(query: str) -> bool:
    """Check if query is a greeting or chit-chat"""
    query_lower = normalize_query(query)
    if query_lower in GREETING_RESPONSES:
        return True
    return False

def get_greeting_response(query: str) -> str | None:
    """Get cached greeting response"""
    query_lower = normalize_query(query)
    return GREETING_RESPONSES.get(query_lower)

def get_simple_response(query: str) -> str | None:
    """Get a deterministic reply for very common simple prompts."""
    query_lower = normalize_query(query)
    return SIMPLE_RESPONSES.get(query_lower)

def should_use_rag(query: str) -> bool:
    """Determine if we should use RAG for this query"""
    query_lower = query.lower().strip()
    
    # Skip RAG for greetings
    if is_greeting(query):
        return False
    
    # Skip RAG for very short queries (less than 5 chars)
    if len(query_lower) < 5:
        return False
    
    # SIRDC-related keywords that should trigger RAG
    sirdc_keywords = {
        "sirdc", "sir dc", "scientific", "industrial", "research", 
        "development", "centre", "center", "cluster", "training", 
        "course", "product", "service", "agriculture", "energy", 
        "power", "health", "mining", "mineral", "ict", "water", 
        "environment", "built environment", "transport", "commercialisation",
        "zts", "foundry", "seed", "potato", "maize", "vaccine",
        "metrology", "calibration", "biotechnology", "food", "nutrition",
        "sme", "enterprise", "partnership", "director", "ceo",
        "executive", "board", "chairperson", "manager"
    }
    
    # Check if query contains any SIRDC keyword
    for keyword in sirdc_keywords:
        if keyword in query_lower:
            return True
    
    # For queries longer than 20 chars that don't have keywords, use RAG anyway
    return len(query_lower) > 20

# ============================================
# BUILD VECTORSTORE FROM SIRD DATASET
# ============================================
def build_vectorstore_from_data():
    """Build FAISS index from SIRD documents data"""
    if faiss is None or SentenceTransformer is None:
        print("⚠️ FAISS or SentenceTransformer not available. RAG disabled.")
        return None, None, None
    
    try:
        print("🔄 Building SIRD vectorstore...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        texts = [doc["text"] for doc in SIRD_DOCUMENTS]
        embeddings = model.encode(texts, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # Save metadata
        meta = []
        for doc in SIRD_DOCUMENTS:
            meta.append({
                "text": doc["text"],
                "source": doc.get("source", "unknown")
            })
        
        print(f"✅ Vectorstore built with {len(SIRD_DOCUMENTS)} SIRD documents!")
        return index, meta, model
        
    except Exception as e:
        print(f"❌ Error building vectorstore: {e}")
        return None, None, None

# ============================================
# INITIALIZE VECTORSTORE
# ============================================
# First try to load from disk
VECTOR_DIR = BASE_DIR / "vectorstore"
INDEX = None
META = None
EMBED_MODEL = None
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Try to load from disk first (from ingest.py)
if faiss and SentenceTransformer and VECTOR_DIR.exists() and (VECTOR_DIR / "index.faiss").exists():
    try:
        print("🔄 Loading SIRD vectorstore from disk...")
        INDEX = faiss.read_index(str(VECTOR_DIR / "index.faiss"))
        with open(VECTOR_DIR / "meta.json", "r", encoding="utf-8") as f:
            META = json.load(f)
        EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
        print(f"✅ Loaded SIRD vectorstore with {len(META)} documents!")
    except Exception as e:
        print(f"❌ Error loading vectorstore: {e}")
        INDEX, META, EMBED_MODEL = build_vectorstore_from_data()
else:
    print("📁 No vectorstore found on disk, building from SIRD dataset...")
    INDEX, META, EMBED_MODEL = build_vectorstore_from_data()


@app.get("/")
async def serve_frontend():
    return FileResponse(BASE_DIR / "index.html")


# Map to store connected users: { "username": WebSocket }
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_personal_message(self, message: dict, user_id: str):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_json(message)


manager = ConnectionManager()


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(user_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            target_user = data.get("target")
            message_type = data.get("type")
            payload = data.get("payload")
            await manager.send_personal_message(
                {"from": user_id, "type": message_type, "payload": payload},
                target_user,
            )
    except WebSocketDisconnect:
        manager.disconnect(user_id)


# ============================================
# FIXED: _call_local_model with retry logic
# ============================================
async def _call_local_model(
    prompt: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    model_override: str | None = None,
) -> str:
    def select_preferred_model(model_names: list[str]) -> str | None:
        normalized = [name.lower() for name in model_names if name]
        for preferred in PREFERRED_OLLAMA_MODELS:
            for name in normalized:
                if name == preferred or name.startswith(f"{preferred}:"):
                    return name
        for name in normalized:
            if name.startswith("llama"):
                return name
        return None

    # Use the model from env or override
    model_name = model_override or os.getenv("OLLAMA_MODEL")
    
    # If still None, try to auto-detect
    if not model_name:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                tags_resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if tags_resp.status_code == 200:
                tags_data = tags_resp.json()
                models = tags_data.get("models", [])
                if models and len(models) > 0:
                    model_names = [m.get("name") for m in models if m.get("name")]
                    model_name = select_preferred_model(model_names)
                    print(f"✅ Auto-detected model: {model_name}")
        except Exception as e:
            print(f"⚠️ Could not auto-detect model: {e}")
    
    # Final fallback
    if not model_name:
        model_name = "llama3.2:3b"
        print(f"ℹ️ Using fallback model: {model_name}")

    # Build the payload with smaller defaults
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens or 150,  # Limit output length
            "temperature": temperature or 0.7,
        }
    }

    print(f"📤 Sending to Ollama: model={model_name}, prompt_length={len(prompt)}")
    
    def fallback_reply(reason: str) -> str:
        print(f"⚠️ Falling back to safe reply: {reason}")
        return (
            "I’m having trouble reaching the local model right now, but I’m still here. "
            "Please try again in a moment, or ask me a SIRDC question and I’ll help with what I can."
        )

    # Retry logic with 3 attempts and shorter timeout
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:  # keep requests responsive
                resp = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
                
                if resp.status_code != 200:
                    print(f"❌ Ollama error: {resp.status_code} - {resp.text}")
                    if attempt < max_retries - 1:
                        print(f"🔄 Retrying... (attempt {attempt + 2}/{max_retries})")
                        await asyncio.sleep(2)  # Wait 2 seconds before retry
                        continue
                    return fallback_reply(f"ollama returned {resp.status_code}")
                
                data = resp.json()
                print(f"✅ Ollama response received")
                
                if isinstance(data, dict):
                    if "response" in data:
                        return data["response"]
                    if "message" in data and isinstance(data["message"], dict):
                        return data["message"].get("content") or str(data["message"])
                
                return str(data)
                
        except httpx.TimeoutException:
            print(f"⏱️ Ollama timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                print(f"🔄 Retrying... (attempt {attempt + 2}/{max_retries})")
                await asyncio.sleep(2)  # Wait 2 seconds before retry
                continue
            print("❌ Ollama request timed out after all retries")
            return fallback_reply("request timed out")
        except Exception as e:
            print(f"❌ Ollama request failed: {e}")
            if attempt < max_retries - 1:
                print(f"🔄 Retrying... (attempt {attempt + 2}/{max_retries})")
                await asyncio.sleep(2)
                continue
            return fallback_reply(str(e))
    
    return fallback_reply("all attempts failed")


# ============================================
# FIXED: retrieve_context with reduced size
# ============================================
def retrieve_context(query: str, top_k: int = 2):  # REDUCED from 3 to 2
    """Retrieve relevant context with chunk limiting"""
    if INDEX is None or EMBED_MODEL is None or META is None:
        return []
    try:
        q_emb = EMBED_MODEL.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = INDEX.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            try:
                if idx < len(META):
                    text = META[idx]["text"]
                    # Limit each chunk to 300 characters (REDUCED from 500)
                    if len(text) > 300:
                        text = text[:300] + "..."
                    results.append(text)
            except Exception:
                continue
        return results
    except Exception as e:
        print(f"⚠️ Error retrieving context: {e}")
        return []


@app.post("/api/chat")
async def chat_endpoint(req: Request):
    try:
        payload = await req.json()
        text = payload.get("text")
        if not text:
            raise HTTPException(status_code=400, detail='Missing "text" in request body')

        # Check for greeting responses first (no Ollama call)
        greeting_response = get_greeting_response(text)
        if greeting_response:
            print(f"📝 Greeting detected: '{text}' - returning cached response")
            return JSONResponse({"reply": greeting_response, "retrieved": 0})

        simple_response = get_simple_response(text)
        if simple_response:
            print(f"📝 Simple prompt detected: '{text}' - returning cached response")
            return JSONResponse({"reply": simple_response, "retrieved": 0})

        # SIRD system instruction - shorter version
        SIRD_SYSTEM_INSTRUCTION = "You are RALPH, an assistant for SIRDC (Scientific and Industrial Research and Development Centre). Answer questions about SIRDC's clusters, training, products, and services concisely."
        
        system = (
            payload.get("system")
            or os.getenv("DEFAULT_SYSTEM_PROMPT")
            or SIRD_SYSTEM_INSTRUCTION
        )
        temperature = payload.get("temperature")
        max_tokens = payload.get("max_tokens")
        model_override = payload.get("model")

        # Only use RAG if needed
        context_blocks = []
        context_text = ""
        
        if should_use_rag(text) and INDEX is not None:
            top_k = int(payload.get("top_k", os.getenv("DEFAULT_TOP_K", 2)))  # REDUCED to 2
            context_blocks = retrieve_context(text, top_k=top_k)
            
            if context_blocks:
                context_text = "\n\n---\n\n".join(context_blocks)
                # Limit total context to 500 characters (REDUCED from 800)
                if len(context_text) > 500:
                    context_text = context_text[:500] + "..."
                print(f"📚 Retrieved {len(context_blocks)} context chunks")
            else:
                print("📚 No context retrieved")
        else:
            print("📝 Skipping RAG for simple query")

        # Build prompt - simpler format
        if context_text:
            prompt = f"{system}\n\nContext:\n{context_text}\n\nUser: {text}\n\nAssistant:"
        else:
            prompt = f"{system}\n\nUser: {text}\n\nAssistant:"
        
        print(f"📤 Sending prompt to Ollama (length: {len(prompt)})")
        
        reply = await _call_local_model(
            prompt,
            temperature=temperature or 0.7,
            max_tokens=max_tokens or 150,  # REDUCED from 200 to 150
            model_override=model_override,
        )
        return JSONResponse({"reply": reply, "retrieved": len(context_blocks)})
    except Exception as e:
        print(f"❌ Error in chat_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))