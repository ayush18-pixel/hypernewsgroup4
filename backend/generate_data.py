"""
generate_data.py
Generates/loads the news dataset, computes embeddings, builds and saves the
FAISS index to disk so subsequent backend startups are instant.
"""
import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR   = os.path.join(_BASE, "data")
PARQUET    = os.path.join(DATA_DIR, "news_processed.parquet")
EMB_FILE   = os.path.join(DATA_DIR, "article_embeddings.npy")
FAISS_FILE = os.path.join(DATA_DIR, "news_faiss.index")

ARTICLES = [
    # ── TECHNOLOGY ──────────────────────────────────────────────────────────
    {"id": "T001", "category": "Technology", "title": "OpenAI Releases GPT-5 with 1 Trillion Parameters", "abstract": "OpenAI has unveiled GPT-5, its most powerful language model yet, boasting 1 trillion parameters and unprecedented reasoning capabilities that outperform human experts in most benchmarks."},
    {"id": "T002", "category": "Technology", "title": "Apple Vision Pro 2 Arrives with Holographic Display", "abstract": "Apple's second-generation Vision Pro headset introduces true holographic projection, spatial computing apps, and a lighter form factor aimed at mass-market adoption."},
    {"id": "T003", "category": "Technology", "title": "Google DeepMind Achieves AGI Milestone in Protein Research", "abstract": "DeepMind researchers announced a major breakthrough after their AI system autonomously designed novel proteins that defeated antibiotic-resistant bacteria in lab trials."},
    {"id": "T004", "category": "Technology", "title": "Meta Launches Llama 4 as Open-Source Foundation Model", "abstract": "Meta released Llama 4 with a permissive license, enabling developers worldwide to build custom AI applications without expensive API calls or proprietary restrictions."},
    {"id": "T005", "category": "Technology", "title": "Quantum Computing Startup Claims 10,000-Qubit Processor", "abstract": "A Silicon Valley startup demonstrated a 10,000-qubit quantum processor that can break current RSA encryption standards, prompting urgent calls for post-quantum cryptography."},
    {"id": "T006", "category": "Technology", "title": "Nvidia Blackwell B300 GPUs Shatter AI Training Records", "abstract": "Nvidia's latest Blackwell B300 architecture delivers 40 exaflops of AI performance per rack, making large model training 10x faster than previous generations."},
    {"id": "T007", "category": "Technology", "title": "Anthropic Claude 4 Achieves 99% Accuracy on Math Olympiad", "abstract": "Anthropic published results showing Claude 4 achieving near-perfect scores on International Mathematical Olympiad problems, surpassing human gold medalists for the first time."},
    {"id": "T008", "category": "Technology", "title": "SpaceX Starlink Gen 3 Promises 10 Gbps Rural Internet", "abstract": "SpaceX launched 300 new Starlink satellites equipped with laser inter-links enabling 10 Gbps download speeds and under 10ms latency for rural and remote communities globally."},
    {"id": "T009", "category": "Technology", "title": "Microsoft Copilot+ Embedded in Every Windows App", "abstract": "Microsoft rolled out a sweeping update embedding Copilot+ AI directly into File Explorer, Notepad, and all Office suite apps, transforming how users interact with their PCs."},
    {"id": "T010", "category": "Technology", "title": "AI-Powered Code Generation Replaces Junior Developers at Scale", "abstract": "A new study found Fortune 500 companies reduced junior engineering hiring by 35% after deploying AI coding agents like GitHub Copilot Workspace for autonomous feature development."},
    {"id": "T011", "category": "Technology", "title": "Samsung Unveils Foldable Phone That Rolls Into a Bracelet", "abstract": "Samsung's Galaxy Roll concept phone transforms from a 6-inch display into a wearable wristband, using flexible OLED and shape-memory alloy hinges."},
    {"id": "T012", "category": "Technology", "title": "Brain-Computer Interface Startup Raises $500M Series B", "abstract": "Synchron, a competitor to Neuralink, raised $500 million to expand clinical trials of its minimally invasive BCI that lets paralyzed patients control computers with thoughts."},
    {"id": "T013", "category": "Technology", "title": "Open-Source AI Image Model Surpasses Midjourney Quality", "abstract": "Stability AI released Stable Diffusion XL Turbo 2.0 generating photorealistic images indistinguishable from professional photography in under 0.5 seconds."},
    {"id": "T014", "category": "Technology", "title": "Electric Vehicle Battery Charges to 80% in 5 Minutes", "abstract": "A startup from MIT developed a solid-state battery pack using lithium-ceramic electrodes that sustain ultra-fast charging without degradation for 1 million cycles."},
    {"id": "T015", "category": "Technology", "title": "AI Startup Funding Hits Record $120 Billion in Q1 2026", "abstract": "Venture capital investment in AI companies reached an all-time quarterly high of $120 billion globally in Q1 2026, with most capital flowing into foundation model and AI infrastructure plays."},
    {"id": "T016", "category": "Technology", "title": "GitHub Copilot Workspace Now Builds Full Applications Autonomously", "abstract": "GitHub's Copilot Workspace upgrade allows developers to describe a feature in plain English and have the AI autonomously write, test, and open a pull request for the full implementation."},
    {"id": "T017", "category": "Technology", "title": "Boston Dynamics Ships 10,000 Atlas Humanoid Robots to Amazon", "abstract": "Boston Dynamics fulfilled its first commercial order of 10,000 Atlas humanoid robots to Amazon warehouses, with the robots handling 95% of package sorting autonomously."},
    {"id": "T018", "category": "Technology", "title": "5G Satellite Coverage Now Reaches Every Point on Earth", "abstract": "The last gap in global 5G satellite coverage was closed this week, meaning mobile connectivity is now available at the North Pole, ocean depths, and the most remote deserts."},
    # ── SPORTS ──────────────────────────────────────────────────────────────
    {"id": "S001", "category": "Sports", "title": "India Wins ICC Cricket World Cup in a Thrilling Final", "abstract": "India claimed the ICC Cricket World Cup defeating Australia by 6 runs in a nail-biting final at Mumbai's Wankhede Stadium, with Virat Kohli scoring a historic century."},
    {"id": "S002", "category": "Sports", "title": "Messi Announces Retirement After Record 10th Ballon d'Or", "abstract": "Lionel Messi announced his retirement from professional football after receiving an unprecedented tenth Ballon d'Or award, capping the greatest career in the sport's history."},
    {"id": "S003", "category": "Sports", "title": "Serena Williams Returns to Win US Open at Age 46", "abstract": "In a stunning sporting comeback, Serena Williams won the US Open women's singles title at 46, defeating the world number one in a historic three-set final."},
    {"id": "S004", "category": "Sports", "title": "NBA Finals: Golden State Warriors Claim 8th Championship", "abstract": "The Golden State Warriors defeated the Boston Celtics 4-2 in the NBA Finals, with Stephen Curry delivering a 50-point performance in the decisive Game 6."},
    {"id": "S005", "category": "Sports", "title": "Real Madrid Wins Champions League for Record 20th Time", "abstract": "Real Madrid secured their record 20th UEFA Champions League trophy with a stunning comeback in the final against Manchester City, scoring twice in added time."},
    {"id": "S006", "category": "Sports", "title": "Djokovic Breaks All-Time Grand Slam Record with 30th Title", "abstract": "Novak Djokovic cemented his GOAT status by winning Wimbledon for the 10th time, bringing his Grand Slam total to 30 and extending his record beyond all historical rivals."},
    {"id": "S007", "category": "Sports", "title": "Formula 1 Introduces Electric Hybrid Season Starting 2027", "abstract": "F1 announced that starting 2027, all cars must run on fully electric power for 30% of each race lap, fundamentally changing team strategies and circuit designs."},
    {"id": "S008", "category": "Sports", "title": "AI Referee System Eliminates Controversial Calls in Premier League", "abstract": "The Premier League's new AI referee system using 32 cameras and computer vision made zero controversial offside or foul decisions in its first full season of deployment."},
    {"id": "S009", "category": "Sports", "title": "Shohei Ohtani Wins 4th MVP as Both Pitcher and Hitter", "abstract": "Shohei Ohtani won his fourth AL MVP award after posting a .340 batting average with 55 home runs while also going 22-3 as a starting pitcher with a 1.8 ERA."},
    {"id": "S010", "category": "Sports", "title": "E-Sports Tournament Prize Pool Surpasses $500 Million", "abstract": "The Dota 2 International 2026 prize pool crossed $500 million, exceeding the combined prize money of Wimbledon and the US Open for the first time in history."},
    {"id": "S011", "category": "Sports", "title": "Women's FIFA World Cup Draws 2 Billion Viewers Globally", "abstract": "The 2026 Women's FIFA World Cup final between Spain and Brazil was watched by 2 billion people globally, surpassing the men's tournament viewership for the first time."},
    {"id": "S012", "category": "Sports", "title": "Pole Vault World Record Broken by 30cm at Indoor Championships", "abstract": "Armand Duplantis shattered his own pole vault world record by a remarkable 30 centimeters at the World Indoor Athletics Championships in Vienna."},
    # ── POLITICS ────────────────────────────────────────────────────────────
    {"id": "P001", "category": "Politics", "title": "UN Security Council Passes Historic Climate Emergency Resolution", "abstract": "All 15 UN Security Council members unanimously passed a binding climate emergency resolution, requiring nations to achieve net-zero emissions by 2035 or face economic sanctions."},
    {"id": "P002", "category": "Politics", "title": "EU Launches Universal Basic Income Pilot for 10 Million Citizens", "abstract": "The European Union launched a €200 billion pilot program providing €1,500 monthly to 10 million citizens across 12 member states to combat AI-driven job displacement."},
    {"id": "P003", "category": "Politics", "title": "US Congress Passes Comprehensive AI Regulation Act", "abstract": "The US Congress passed the landmark AI Safety and Accountability Act, requiring all AI systems affecting more than 1 million people to undergo mandatory safety audits."},
    {"id": "P004", "category": "Politics", "title": "India-China Border Agreement Ends 5-Year Standoff", "abstract": "India and China signed a comprehensive border management agreement that demilitarizes contested Himalayan territories, ending a five-year military standoff."},
    {"id": "P005", "category": "Politics", "title": "G20 Nations Agree on Global Wealth Tax for Billionaires", "abstract": "G20 finance ministers reached a historic agreement on a 2% annual global minimum wealth tax on individuals with assets exceeding $1 billion."},
    {"id": "P006", "category": "Politics", "title": "Scotland Referendum: 58% Vote for Independence", "abstract": "Scotland voted 58% in favour of independence from the United Kingdom in a legally binding referendum, setting the stage for complex negotiations over assets and currency."},
    {"id": "P007", "category": "Politics", "title": "Brazil Elects First Indigenous President in Historic Election", "abstract": "Brazil elected its first indigenous president, who has pledged to halt all Amazon deforestation, renegotiate oil contracts, and strengthen indigenous land rights."},
    {"id": "P008", "category": "Politics", "title": "NATO Expands to 40 Members with Entry of Five New Nations", "abstract": "NATO welcomed five new member states including Georgia, Kosovo, and three Central Asian republics, expanding the alliance to 40 nations amid ongoing tensions with Russia."},
    {"id": "P009", "category": "Politics", "title": "Global Plastics Treaty Signed by 196 Nations in Nairobi", "abstract": "All 196 UN member states signed the Global Plastics Treaty in Nairobi, committing to a 90% reduction in virgin plastic production by 2030."},
    {"id": "P010", "category": "Politics", "title": "China Announces Taiwan Economic Integration Framework", "abstract": "China proposed a comprehensive economic integration framework for Taiwan, offering zero tariffs and shared infrastructure investment as an alternative to political unification pressure."},
    # ── ENTERTAINMENT ────────────────────────────────────────────────────────
    {"id": "E001", "category": "Entertainment", "title": "Oscar-Winning AI-Directed Film Sparks Industry Debate", "abstract": "An AI-directed film won the Academy Award for Best Picture, triggering fierce debate about creative authorship, SAG-AFTRA intellectual property rights, and the future of Hollywood."},
    {"id": "E002", "category": "Entertainment", "title": "Taylor Swift Eras Tour 2026 Breaks Attendance Records", "abstract": "Taylor Swift's second Eras Tour grossed $3 billion worldwide, playing to over 20 million fans across 140 shows and setting records for the highest-grossing concert tour in history."},
    {"id": "E003", "category": "Entertainment", "title": "Netflix Releases Interactive Series Where AI Adapts Storyline", "abstract": "Netflix debuted an experimental interactive series where a generative AI adapts the plot, dialogue, and character relationships in real time based on the viewer's emotional responses."},
    {"id": "E004", "category": "Entertainment", "title": "BTS Reunites After Military Service for World Stadium Tour", "abstract": "BTS announced a reunion world stadium tour following all members' completion of mandatory South Korean military service, with 50 dates selling out in under 8 minutes."},
    {"id": "E005", "category": "Entertainment", "title": "Marvel Multiverse Saga Finale Earns $3 Billion Opening Weekend", "abstract": "The Avengers: Multiverse Saga Finale earned $3 billion in its global opening weekend, shattering every box office record and becoming the first film to cross $10 billion total gross."},
    {"id": "E006", "category": "Entertainment", "title": "Spotify AI DJ Luna Becomes Most-Followed Artist on Platform", "abstract": "Spotify's AI DJ Luna surpassed 500 million followers, making its AI personality the most followed entity on the streaming platform."},
    {"id": "E007", "category": "Entertainment", "title": "The Last of Us Season 3 Wins Emmy for Best Drama", "abstract": "HBO's The Last of Us Season 3 swept the Emmy Awards, winning Best Drama, Best Director, and Best Actor, cementing its status as the greatest video game adaptation ever made."},
    {"id": "E008", "category": "Entertainment", "title": "Virtual Concert in Metaverse Draws 50 Million Concurrent Viewers", "abstract": "Ariana Grande's metaverse concert on virtual platform Horizon Worlds set a record with 50 million simultaneous viewers, with fans purchasing 8 million virtual merchandise items."},
    {"id": "E009", "category": "Entertainment", "title": "Rihanna Returns with First Album in 10 Years", "abstract": "Rihanna released her long-awaited ninth studio album 'Reign', which broke streaming records for first-week plays and debuted at number one in 78 countries simultaneously."},
    {"id": "E010", "category": "Entertainment", "title": "Video Game Horizon Infinite Ships 50 Million Copies in First Week", "abstract": "Guerrilla Games' Horizon Infinite sold 50 million copies across PlayStation and PC in its debut week, breaking the entertainment industry's all-time launch sales record."},
    # ── SCIENCE ─────────────────────────────────────────────────────────────
    {"id": "SC001", "category": "Science", "title": "Researchers Reverse Aging in Human Cells by 25 Years", "abstract": "Harvard scientists demonstrated reversing epigenetic aging markers in human liver cells by 25 years using Yamanaka factor gene therapy, with clinical trials planned for 2027."},
    {"id": "SC002", "category": "Science", "title": "James Webb Telescope Discovers Biosignature on Exoplanet", "abstract": "NASA's James Webb Space Telescope detected dimethyl sulfide — a chemical exclusively produced by life on Earth — in the atmosphere of exoplanet K2-18b, 120 light-years away."},
    {"id": "SC003", "category": "Science", "title": "CRISPR Therapy Cures Sickle Cell Disease in 1,000 Patients", "abstract": "The first 1,000 patients treated with CRISPR gene-editing therapy for sickle cell disease show complete remission after five years, with no serious adverse events reported."},
    {"id": "SC004", "category": "Science", "title": "Nuclear Fusion Reactor Achieves Net Energy Gain for 10 Hours", "abstract": "The International Thermonuclear Experimental Reactor achieved sustained net energy gain for 10 continuous hours, a critical milestone toward commercially viable fusion power by 2035."},
    {"id": "SC005", "category": "Science", "title": "Scientists Grow Functional Human Kidney in Pig Host", "abstract": "University of Tokyo researchers grew a fully functional human kidney inside a pig using personalized stem cells and CRISPR, paving the way to end transplant organ shortages."},
    {"id": "SC006", "category": "Science", "title": "Dark Matter Detected for First Time in Laboratory Conditions", "abstract": "Physicists at CERN announced the first laboratory detection of dark matter particles using a novel axion detector cooled to near absolute zero, confirming decades of cosmological models."},
    {"id": "SC007", "category": "Science", "title": "mRNA Cancer Vaccine Shows 93% Remission in Pancreatic Trials", "abstract": "Moderna's personalized mRNA cancer vaccine combined with immunotherapy achieved a 93% remission rate in pancreatic cancer patients, one of the deadliest forms of the disease."},
    {"id": "SC008", "category": "Science", "title": "Ocean Plastic-Eating Enzyme Deployed at Scale in Pacific", "abstract": "An engineered plastic-eating enzyme was released in 500 ocean locations in the Pacific, breaking down 10,000 tonnes of PET plastic per month with no ecological side effects."},
    {"id": "SC009", "category": "Science", "title": "Neuralink Patient Controls Robotic Arm With Thought Alone", "abstract": "A quadriplegic patient with Neuralink's brain implant successfully controlled a six-degree-of-freedom robotic arm with enough precision to play piano, write, and pour liquids."},
    {"id": "SC010", "category": "Science", "title": "New Antibiotic Kills Every Known Drug-Resistant Bacteria Strain", "abstract": "Scientists discovered Clovibactin 2.0, a new class of antibiotic derived from soil bacteria that kills 100% of tested drug-resistant strains without triggering evolutionary resistance."},
    # ── BUSINESS ────────────────────────────────────────────────────────────
    {"id": "B001", "category": "Business", "title": "Apple Becomes First $10 Trillion Market Cap Company", "abstract": "Apple Inc. crossed the $10 trillion market capitalization milestone, driven by record iPhone 17 sales, services growth, and investor optimism about its generative AI integration."},
    {"id": "B002", "category": "Business", "title": "Amazon Acquires Shopify for $200 Billion in All-Stock Deal", "abstract": "Amazon announced the acquisition of Shopify for $200 billion, creating a dominant e-commerce platform that handles 60% of global online retail transactions."},
    {"id": "B003", "category": "Business", "title": "Global Recession Fears Rise as Manufacturing Output Falls 8%", "abstract": "The IMF issued a recession warning after global manufacturing output fell 8% in Q1, citing overcapacity from AI automation, weak consumer demand, and tightening credit conditions."},
    {"id": "B004", "category": "Business", "title": "Harvey AI Raises $1 Billion to Build AI-Powered Legal Platform", "abstract": "Harvey AI raised $1 billion at a $15 billion valuation to build AI legal services that can draft contracts, conduct discovery, and represent clients in routine legal proceedings."},
    {"id": "B005", "category": "Business", "title": "Goldman Sachs Replaces 40% of Analysts with AI Systems", "abstract": "Goldman Sachs disclosed it replaced 40% of its analyst workforce with AI agents that produce financial models, research reports, and trading strategies at 100x the speed."},
    {"id": "B006", "category": "Business", "title": "India GDP Surpasses Japan to Become World's Third Largest Economy", "abstract": "India's GDP crossed $6.5 trillion, officially surpassing Japan to become the world's third-largest economy, driven by IT services exports, manufacturing growth, and domestic consumption."},
    {"id": "B007", "category": "Business", "title": "Bitcoin Hits $500,000 as Institutional ETF Inflows Surge", "abstract": "Bitcoin crossed the $500,000 mark after record institutional ETF inflows of $50 billion in a single quarter, with BlackRock's Bitcoin ETF becoming the fastest-growing fund in history."},
    {"id": "B008", "category": "Business", "title": "xAI Grok-3 Enterprise Exceeds $10 Billion Revenue", "abstract": "xAI's enterprise API for Grok-3 exceeded $10 billion in annualized revenue six months after launch, making it the fastest-growing B2B AI service in business history."},
    {"id": "B009", "category": "Business", "title": "New York Stock Exchange to Operate 24/7 Starting 2027", "abstract": "The NYSE announced 24/7 trading starting January 2027, driven by AI market-making systems that eliminate the need for human-paced settlement cycles and time-zone restrictions."},
    {"id": "B010", "category": "Business", "title": "Remote Work Becomes Permanent for 60% of White-Collar Jobs", "abstract": "A global survey found 60% of white-collar employers now offer permanent fully-remote arrangements, fundamentally altering commercial real estate and urban planning."},
    {"id": "B011", "category": "Business", "title": "EV Market Share Crosses 50% of New Car Sales Globally", "abstract": "Electric vehicles accounted for 50.3% of all new car sales globally in Q1 2026, with China leading at 78%, Europe at 62%, and the United States at 40%."},
    # ── LIFESTYLE ────────────────────────────────────────────────────────────
    {"id": "L001", "category": "Lifestyle", "title": "Mindfulness Apps Report 500% Surge in Teen Usage", "abstract": "Mental wellness applications like Calm and Headspace reported a 500% surge in teenage users following school district mandates for daily five-minute mindfulness sessions."},
    {"id": "L002", "category": "Lifestyle", "title": "Plant-Based Diet Reduces Heart Disease Risk by 32%", "abstract": "A 20-year longitudinal study of 80,000 participants found those following whole-food plant-based diets had a 32% lower risk of cardiovascular disease compared to omnivores."},
    {"id": "L003", "category": "Lifestyle", "title": "Digital Detox Retreats Become the Fastest-Growing Travel Category", "abstract": "No-screens, no-Wi-Fi digital detox retreats grew 180% in bookings in 2026, with waitlists extending to 18 months at premium destinations in Bhutan, Iceland, and Patagonia."},
    {"id": "L004", "category": "Lifestyle", "title": "Gen Z Drives 40% Decline in Alcohol Consumption Globally", "abstract": "Global alcohol consumption fell 40% over the past decade, driven by Gen Z's preference for non-alcoholic beverages, mindfulness culture, and cannabis-based social alternatives."},
    {"id": "L005", "category": "Lifestyle", "title": "Home Biohacking Market Hits $50 Billion", "abstract": "The home biohacking market including continuous glucose monitors, IV drips, red light therapy, and cold plunge systems crossed $50 billion as wellness optimization enters everyday households."},
    {"id": "L006", "category": "Lifestyle", "title": "Four-Day Work Week Becomes Mandatory in 12 Countries", "abstract": "Twelve countries including Germany, Japan, and Canada passed legislation mandating a four-day, 32-hour work week with full pay, citing productivity studies showing a 25% output increase."},
    {"id": "L007", "category": "Lifestyle", "title": "Solo Travel Among Women Grows 200% as Safety Apps Improve", "abstract": "Solo travel among women grew 200% following the launch of AI-powered safety applications providing real-time risk assessment, emergency contacts, and verified accommodation networks."},
    {"id": "L008", "category": "Lifestyle", "title": "Sleep Tech Market Booms with Smart Mattresses", "abstract": "Smart mattresses equipped with biometric sensors that automatically adjust firmness, temperature, and position based on sleep stage data become the top-selling furniture category."},
    {"id": "L009", "category": "Lifestyle", "title": "Coffee Alternatives Boom: Mushroom and Adaptogen Drinks Hit $20B", "abstract": "The functional beverage market featuring lion's mane mushroom coffee, ashwagandha lattes, and reishi tea exceeded $20 billion as consumers seek energy without caffeine crash side effects."},
    {"id": "L010", "category": "Lifestyle", "title": "Urban Farming Takes Root: 30% of City Dwellers Grow Own Food", "abstract": "30% of urban residents in major cities now grow some of their own food using vertical garden systems, balcony hydroponics, or community rooftop farms, reducing grocery bills by 25%."},
    # ── HEALTH ──────────────────────────────────────────────────────────────
    {"id": "H001", "category": "Health", "title": "GLP-1 Drug Revolution: Obesity Rate Falls Below 20% in US", "abstract": "The widespread adoption of GLP-1 receptor agonists like semaglutide caused the US obesity rate to fall below 20% for the first time in 40 years, with profound implications for healthcare costs."},
    {"id": "H002", "category": "Health", "title": "AI Diagnostic Tool Detects Cancer 7 Years Earlier Than Current Methods", "abstract": "A multimodal AI system analyzing blood proteomics and genomics detected 12 types of cancer an average of seven years earlier than conventional screening methods across 50,000 patients."},
    {"id": "H003", "category": "Health", "title": "Alzheimer's Drug Halts Disease Progression in 80% of Patients", "abstract": "A new amyloid-clearing monoclonal antibody demonstrated 80% disease progression halt in early-stage Alzheimer's patients across a 3,000-person trial."},
    {"id": "H004", "category": "Health", "title": "WHO Declares Loneliness a Global Public Health Emergency", "abstract": "The World Health Organization declared loneliness a global public health emergency, citing data linking social isolation to mortality risks equivalent to smoking 15 cigarettes daily."},
    {"id": "H005", "category": "Health", "title": "Continuous Health Monitor Ring Replaces Annual Check-Ups", "abstract": "A ring-sized wearable that continuously monitors 47 biomarkers including blood glucose, cortisol, and cardiac biomarkers is replacing annual doctor check-ups for 10 million early adopters."},
    {"id": "H006", "category": "Health", "title": "Universal Mental Health Coverage Passes in 30 Countries", "abstract": "Thirty nations ratified a WHO mental health framework mandating free access to therapy, psychiatric medication, and crisis intervention services as part of universal healthcare coverage."},
    {"id": "H007", "category": "Health", "title": "Gut Microbiome Therapy Reverses Type 2 Diabetes in Clinical Trial", "abstract": "A personalized gut microbiome transplant protocol achieved complete Type 2 diabetes remission in 67% of 800 patients, offering a non-pharmaceutical alternative to lifelong medication."},
    {"id": "H008", "category": "Health", "title": "Burnout Recognition Added to ICD-12 as Diagnosable Condition", "abstract": "The World Health Organization officially classified burnout as a diagnosable medical condition in ICD-12, entitling affected workers to paid medical leave and employer duty of care obligations."},
    {"id": "H009", "category": "Health", "title": "Maternal Mortality Halved Through AI-Assisted Obstetrics", "abstract": "AI-assisted obstetric care systems deployed in 60 low-income countries halved maternal mortality rates by predicting complications 6 hours earlier with 94% accuracy."},
    {"id": "H010", "category": "Health", "title": "Blue Light Glasses Proven Ineffective; Screen Breaks Are the Solution", "abstract": "A comprehensive meta-analysis of 80 studies found blue-light-blocking glasses have no statistically significant effect on digital eye strain, while the 20-20-20 rule reduced symptoms by 55%."},
    {"id": "H011", "category": "Health", "title": "Exercise Pill Shows Same Cardiovascular Benefits as 30-Min Run", "abstract": "A novel AMPK-activating compound replicated the cardiovascular benefits of a 30-minute aerobic workout in sedentary adults in a 12-month randomized controlled trial."},
    {"id": "H012", "category": "Health", "title": "Sleep Deprivation Declared Silent Pandemic by Sleep Foundation", "abstract": "The Global Sleep Foundation released data showing 45% of the global population chronically sleeps fewer than 6 hours nightly, linked to rising rates of obesity, dementia, and immune dysfunction."},
]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Build DataFrame ──────────────────────────────────────────────────────
    print("📰 Preparing news dataset...")
    df = pd.DataFrame(ARTICLES).rename(columns={"id": "news_id"})
    df["text"] = df["title"] + ". " + df["abstract"]
    df.to_parquet(PARQUET, index=False)
    print(f"   Saved {len(df)} articles → {PARQUET}")

    # ── Embeddings ───────────────────────────────────────────────────────────
    print("🔢 Computing embeddings (this takes ~30s on CPU)...")
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["text"].tolist(), batch_size=32, show_progress_bar=True)
    embeddings = embeddings.astype("float32")
    np.save(EMB_FILE, embeddings)
    print(f"   Saved embeddings {embeddings.shape} → {EMB_FILE}")

    # ── FAISS index ──────────────────────────────────────────────────────────
    print("🔍 Building FAISS index...")
    index   = faiss.IndexFlatIP(embeddings.shape[1])
    emb_copy = embeddings.copy()
    faiss.normalize_L2(emb_copy)
    index.add(emb_copy)
    faiss.write_index(index, FAISS_FILE)
    print(f"   Saved FAISS index → {FAISS_FILE}")

    print("✅ Data pipeline complete!")


if __name__ == "__main__":
    main()
