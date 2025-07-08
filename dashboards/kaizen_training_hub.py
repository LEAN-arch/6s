"""
Renders the Continuous Improvement & Knowledge Hub.

This module serves as a central repository for documenting continuous improvement
(Kaizen) events and for hosting training materials related to quality principles.
It supports the MBB's role in coaching, mentoring, and fostering a culture of
quality by sharing successes and enabling skill development.

SME Overhaul (Definitive All-Inclusive Edition):
- This is the complete, unbridged file, merging the original application logic
  with the massively extended, academic-grade content.
- It uses a hybrid data strategy: it attempts to load live data from the
  SessionStateManager, but falls back to rich, hardcoded data if none is found,
  guaranteeing no "empty state" errors.
- The UI is adaptive, capable of rendering both simple, original data structures
  and the enhanced, detailed content without error.
- All content streams—Kaizen Log, Training Library, Glossary with Formulas, and
  a Foundational Bibliography—are included in their entirety.
"""

import logging
import pandas as pd
import streamlit as st

from six_sigma.data.session_state_manager import SessionStateManager


# ==============================================================================
# --- DEFINITIVE, HARDCODED SHOWCASE CONTENT (FALLBACK DATA) ---
# This section contains the rich, academic-grade content. It serves as a
# fallback if the SessionStateManager does not provide live data, making the
# component self-sufficient and demonstrable.
# ==============================================================================

def get_overhauled_kaizen_data():
    """Generates an expanded and diverse set of Kaizen event data."""
    return [
        {
            "id": "KZN-04", "title": "Invoice Processing Lead Time Reduction", "site": "Corporate HQ", "date": "2025-07-20",
            "problem_background": "The average lead time from invoice receipt to payment approval is 18 days, causing late payment fees and straining supplier relationships. The goal was to reduce this to <5 days by eliminating non-value-added steps.",
            "analysis_and_countermeasures": """
            - **Analysis:** A detailed process map and swimlane diagram revealed significant 'waiting' waste. 80% of the lead time was spent in queues awaiting manual review, data entry into three separate systems, and manager approval.
            - **Countermeasures Implemented:**
                1.  **Eliminated Redundant Data Entry:** Utilized robotic process automation (RPA) to sync data between systems after initial entry.
                2.  **Established Standard Work:** Created a clear policy for approval thresholds, empowering clerks to approve payments below $5,000 without manager sign-off.
                3.  **Visual Management:** Implemented a digital Kanban board (To Do, In Progress, Done) for full visibility of the invoice workload.
            """,
            "quantified_results": "Reduced average lead time from 18 days to 3.5 days. Eliminated all late payment fees in the first quarter post-implementation, saving an estimated $120k annually.",
            "key_insight": "Lean principles are not just for the factory floor. Transactional processes are often filled with the most 'hidden' waste, offering huge opportunities for improvement."
        },
        {
            "id": "KZN-03", "title": "Root Cause Analysis (RCA) of Intermittent Sensor Failures", "site": "Shanghai, CN", "date": "2025-07-01",
            "problem_background": "The final test stage for the Affiniti Ultrasound system was experiencing a 4% failure rate due to intermittent 'Signal Lost' errors from a key pressure sensor, causing costly rework and diagnostic time.",
            "analysis_and_countermeasures": """
            - **Analysis:** An Ishikawa (Fishbone) diagram was used to brainstorm potential causes. The '5 Whys' technique was then applied to the most likely cause, 'Incorrect Connector Seating'.
                1. **Why?** The connector was not fully seated.
                2. **Why?** The operator could not get enough leverage.
                3. **Why?** The access angle was awkward.
                4. **Why?** A new bracket was installed in a previous update.
                5. **Why? (Root Cause)** The bracket design did not account for tool clearance for the sensor connector.
            - **Countermeasure Implemented:** A cross-functional team of engineering and manufacturing redesigned the bracket with an access cutout. A torque-limiting screwdriver with an audible 'click' was also introduced as a Poka-Yoke.
            """,
            "quantified_results": "Reduced the specific 'Signal Lost' failure rate from 4% to 0.1% within one week of implementation. Rework costs were reduced by an estimated $250k annually.",
            "key_insight": "Technical problems are often symptoms of process or design flaws. Persistently asking 'Why' moves the team beyond blaming components or people to fixing the underlying system."
        },
        {
            "id": "KZN-02", "title": "SMED on Stamping Press P-101", "site": "Andover, US", "date": "2025-06-15",
            "problem_background": "The Stamping Press P-101 has an average changeover time of 55 minutes, causing significant production downtime and limiting our ability to run smaller, more flexible batch sizes. This fails to meet the operational target of <15 minutes.",
            "analysis_and_countermeasures": """
            - **Analysis:** A Gemba walk and video analysis (based on Shigeo Shingo's SMED methodology) revealed that 70% of changeover activities were 'internal' (machine stopped). Key opportunities included pre-staging dies, standardizing tools, and eliminating manual adjustments.
            - **Countermeasures Implemented:**
                1.  **Converted Internal to External:** Designed a pre-heating cart for the next die set.
                2.  **Standardized Clamping:** Replaced multi-size bolts with standardized, quick-release clamps.
                3.  **Introduced Poka-Yoke:** Added alignment pins to the die-set to eliminate measurement adjustments.
                4.  **Created Standard Work:** Developed a one-page visual guide for the 2-person changeover team.
            """,
            "quantified_results": "Reduced average changeover time from 55 minutes to 9 minutes (an 83% reduction). This unlocked an additional 120 minutes of production capacity per day and enabled an immediate move to a 'pull' system for downstream assembly.",
            "key_insight": "The biggest gains came not from operators working faster, but from eliminating entire steps of the process. True efficiency is in the design of the work itself, not the effort of the worker."
        },
        {
            "id": "KZN-01", "title": "5S Implementation in Main Assembly Cell", "site": "Eindhoven, NL", "date": "2025-05-22",
            "problem_background": "The Main Assembly cell was experiencing frequent micro-stoppages due to operators searching for tools, components, and fixtures. This introduced significant variability into the Takt time and was a source of operator frustration.",
            "analysis_and_countermeasures": """
            - **Analysis:** A spaghetti diagram of operator movement during a single shift revealed over 400 meters of unnecessary walking. The root cause was a lack of standardized locations for tools and materials.
            - **Countermeasures Implemented (5S):**
                1.  **Sort:** Red-tagged all non-essential items; 3 skids of clutter were removed.
                2.  **Set in Order:** Created shadow boards for all hand tools. Implemented a color-coded bin system for fasteners.
                3.  **Shine:** Conducted a deep clean and established a daily 5-minute cleaning schedule.
                4.  **Standardize:** Laminated standard work instructions for tool placement and end-of-shift cleanup.
                5.  **Sustain:** Added 5S adherence to the daily Gemba walk checklist and supervisor standard work.
            """,
            "quantified_results": "Eliminated 95% of 'searching' time, reducing average assembly time by 15%. Operator-reported ergonomic strain and frustration decreased significantly, confirmed by a post-event survey (results redacted for privacy).",
            "key_insight": "A clean and organized workplace is not about aesthetics; it is a prerequisite for quality and efficiency. When everything has a place, deviations from standard become immediately visible."
        }
    ]

def get_overhauled_training_data():
    """Generates an expanded, comprehensive training library."""
    return [
        {
            "id": "TRN-101",
            "title": "A3 Thinking: The Art of Problem Solving on a Single Page",
            "type": "eLearning",
            "duration_hr": 2.5,
            "target_audience": "All Salaried Staff",
            "link": "#",
            "icon": "📝",
            "description": "This module explores the Toyota Production System's powerful A3 methodology, which structures problem-solving into a narrative format on a single sheet of A3-sized paper. It is a tool for mentorship, clear communication, and data-driven decision making.",
            "learning_objectives": [
                "Understand the 7 sections of a standard A3 Report.",
                "Frame a problem statement effectively.",
                "Use the PDCA (Plan-Do-Check-Act) cycle within the A3 framework.",
                "Visually communicate root cause analysis and countermeasures."
            ],
            "recommended_reading": "'Managing to Learn' by John Shook"
        },
        {
            "id": "TRN-102",
            "title": "Statistical Process Control (SPC) Masterclass",
            "type": "Workshop Slides",
            "duration_hr": 8.0,
            "target_audience": "Engineers, Technicians",
            "link": "#",
            "icon": "📊",
            "description": "A deep dive into the principles of Dr. W. Edwards Deming and Walter Shewhart. This workshop provides the statistical foundation for understanding process variation, distinguishing between common and special causes, and using control charts to monitor and improve process stability.",
            "learning_objectives": [
                "Calculate control limits for I-MR, Xbar-R, and p-charts.",
                "Interpret control chart signals (e.g., Nelson Rules).",
                "Define and calculate process capability indices (Cp, Cpk).",
                "Understand the relationship between process control and process capability."
            ],
            "recommended_reading": "'Understanding Variation' by Donald J. Wheeler"
        },
        {
            "id": "TRN-104", "title": "Failure Mode and Effects Analysis (FMEA)", "type": "eLearning", "duration_hr": 3.0, "target_audience": "Engineering, R&D, Quality", "link": "#", "icon": "🛡️",
            "description": "Learn to proactively identify and mitigate risks in product and process design. This module teaches the systematic approach of FMEA to anticipate potential failures, assess their impact, and implement robust controls before problems reach the customer.",
            "learning_objectives": ["Distinguish between Design FMEAs and Process FMEAs.", "Calculate Risk Priority Numbers (RPN).", "Develop effective detection and prevention controls.", "Integrate FMEA into the product development lifecycle."],
            "recommended_reading": "'The FMEA Pocket Handbook' by D. H. Stamatis"
        },
        {
            "id": "TRN-105", "title": "Value Stream Mapping (VSM)", "type": "Workshop Slides", "duration_hr": 6.0, "target_audience": "Operations, CI Leads, Management", "link": "#", "icon": "🌊",
            "description": "This workshop teaches you how to see the flow of value and, more importantly, the flow of waste. Learn to create current-state and future-state maps that visualize not just material flow, but information flow, to design truly lean systems from end to end.",
            "learning_objectives": ["Identify a value stream and its product family.", "Calculate key metrics like Lead Time, Process Time, and Process Cycle Efficiency.", "Draw current-state and future-state maps using standard iconography.", "Develop a Kaizen-based implementation plan."],
            "recommended_reading": "'Learning to See' by Mike Rother and John Shook"
        },
        {
            "id": "TRN-103",
            "title": "Leading Kaizen Events: A Facilitator's Guide",
            "type": "PDF Guide",
            "duration_hr": 4.0,
            "target_audience": "MBB, Black Belts, CI Leads",
            "link": "#",
            "icon": "🤝",
            "description": "This guide provides a practical, step-by-step framework for planning, executing, and sustaining a successful week-long Kaizen event. It covers team selection, scoping, daily management, and follow-up activities to ensure that improvements are not only made but also maintained.",
            "learning_objectives": [
                "Develop a compelling Kaizen charter.",
                "Manage team dynamics and engage stakeholders.",
                "Facilitate brainstorming and root cause analysis sessions.",
                "Establish a 30-day follow-up plan to ensure sustainability."
            ],
            "recommended_reading": "'Kaizen: The Key to Japan's Competitive Success' by Masaaki Imai"
        }
    ]

def get_glossary_content():
    """Generates a definitive glossary with mathematical formulas."""
    return {
        "Lean Principles": [
            {"term": "Takt Time", "definition": "The rate at which a finished product needs to be completed to meet customer demand. It is the 'heartbeat' of a lean system.", "formula": r"Takt\ Time = \frac{\text{Available Production Time per Day}}{\text{Customer Demand per Day}}"},
            {"term": "Muda (無駄), Mura (斑), Muri (無理)", "definition": "The '3 M's' of waste in the Toyota Production System. **Muda:** Non-value-added waste. **Mura:** Unevenness or irregularity. **Muri:** Overburdening equipment or operators."},
            {"term": "Jidoka (自働化)", "definition": "Autonomation or 'automation with a human touch.' The principle of designing equipment to stop automatically and signal immediately when a problem occurs, preventing the mass production of defects."},
            {"term": "Heijunka (平準化)", "definition": "Production leveling. The process of smoothing the type and quantity of production over a fixed period. This reduces Mura (unevenness) and minimizes inventory."},
            {"term": "Kanban (看板)", "definition": "A scheduling system for lean manufacturing and just-in-time manufacturing (JIT). It is a visual signal (e.g., a card) that triggers an action, such as replenishing a part."},
        ],
        "Six Sigma Concepts": [
            {"term": "DMAIC & DMADV", "definition": "**DMAIC** is the reactive improvement cycle for existing processes. **DMADV** (Define, Measure, Analyze, Design, Verify) is the proactive 'Design for Six Sigma' (DFSS) methodology for creating new processes or products."},
            {"term": "DPMO (Defects Per Million Opportunities)", "definition": "A key metric for process performance. It represents the number of defects in a process per one million opportunities. A Six Sigma process aims for 3.4 DPMO.", "formula": r"DPMO = \frac{\text{Number of Defects}}{\text{Number of Units} \times \text{Opportunities per Unit}} \times 1,000,000"},
            {"term": "Process Capability (Cp & Cpk)", "definition": "Measures how well a process can produce output relative to customer specifications. **Cp** (Potential) assumes perfect centering. **Cpk** (Actual) accounts for process centering. A value >1.33 is a common minimum target.", "formula": r"C_p = \frac{USL - LSL}{6\sigma} \quad | \quad C_{pk} = \min\left(\frac{USL - \mu}{3\sigma}, \frac{\mu - LSL}{3\sigma}\right)"},
            {"term": "Rolled Throughput Yield (RTY)", "definition": "The probability that a multi-step process will produce a defect-free unit. It is the product of the First Time Yields (FTY) of each process step.", "formula": r"RTY = \prod_{i=1}^{n} FTY_i = FTY_1 \times FTY_2 \times \dots \times FTY_n"},
        ],
        "Statistical & Analytical Methods": [
            {"term": "Gage R&R (Repeatability & Reproducibility)", "definition": "A statistical study to evaluate the precision of a measurement system. **Repeatability** is the variation from the same operator using the same tool. **Reproducibility** is the variation between different operators using the same tool."},
            {"term": "Control Chart (Shewhart Chart)", "definition": "A graph used to study how a process changes over time. It shows a center line for the average, an upper line for the upper control limit, and a lower line for the lower control limit.", "formula": r"UCL/LCL = \bar{\bar{x}} \pm 3 \frac{\bar{R}}{d_2} \quad \text{(for Xbar-R charts)}"},
            {"term": "p-value", "definition": "The probability of obtaining test results at least as extreme as the results actually observed, assuming the null hypothesis (H₀) is correct. A small p-value (typically ≤ 0.05) indicates strong evidence to reject the null hypothesis."},
            {"term": "Confidence Interval", "definition": "A range of values, derived from sample statistics, that is likely to contain the value of an unknown population parameter. A 95% confidence interval means we are 95% confident the true population mean lies within that range."},
            {"term": "Regression Analysis", "definition": "A set of statistical processes for estimating the relationships between a dependent variable (Y) and one or more independent variables (X's). The goal is to create a predictive model.", "formula": r"Y = \beta_0 + \beta_1X_1 + \dots + \beta_nX_n + \epsilon"},
        ],
        "AI/ML for Operations": [
            {"term": "Supervised Learning", "definition": "A type of ML where the model learns from labeled data (e.g., images of parts labeled 'Good' or 'Defective'). Used for classification and regression."},
            {"term": "Unsupervised Learning", "definition": "A type of ML where the model finds hidden patterns in unlabeled data. Used for clustering (e.g., finding distinct failure modes) and anomaly detection."},
            {"term": "Isolation Forest", "definition": "An unsupervised algorithm excellent for anomaly detection. It works by 'isolating' outliers, which are easier to separate from the main data cluster."},
            {"term": "SHAP (SHapley Additive exPlanations)", "definition": "A game-theoretic approach to explain the output of any machine learning model. It explains *why* a model made a specific prediction for a single instance by assigning importance values to each feature."}
        ]
    }

def get_bibliography_content():
    """Generates a definitive, curated bibliography."""
    return {
        "Lean Thinking & Culture": [
            {"title": "The Machine That Changed the World", "author": "James P. Womack, Daniel T. Jones, Daniel Roos", "summary": "The foundational text that introduced Lean production to the Western world. Essential reading to understand the 'Why' behind the entire Lean movement and its profound impact on manufacturing."},
            {"title": "The Toyota Way: 14 Management Principles", "author": "Jeffrey Liker", "summary": "A deep dive into the philosophy behind Toyota's success. It codifies the two pillars: 'Continuous Improvement' and 'Respect for People,' providing a blueprint for building a true learning organization."},
            {"title": "Kaizen: The Key to Japan’s Competitive Success", "author": "Masaaki Imai", "summary": "The classic guide to the practice of Kaizen. It masterfully explains how small, incremental changes, driven by everyone in the organization, lead to monumental results over time."},
        ],
        "Six Sigma & Statistical Rigor": [
            {"title": "Out of the Crisis", "author": "W. Edwards Deming", "summary": "The seminal work from the father of the quality movement. Deming’s 14 points provide a timeless management philosophy focused on systemic thinking, understanding variation, and profound knowledge."},
            {"title": "Understanding Variation: The Key to Managing Chaos", "author": "Donald J. Wheeler", "summary": "Arguably the most accessible and practical book ever written on Statistical Process Control (SPC). It masterfully teaches how to distinguish between common and special cause variation, which is the most critical skill in process improvement."},
            {"title": "The Six Sigma Handbook", "author": "Thomas Pyzdek & Paul A. Keller", "summary": "A comprehensive reference guide covering the entire body of knowledge for Six Sigma. An indispensable desk reference for any serious practitioner, from Green Belt to Master Black Belt."},
        ],
        "Modern Integration & Leadership": [
            {"title": "The Goal: A Process of Ongoing Improvement", "author": "Eliyahu M. Goldratt", "summary": "A business novel that brilliantly teaches the Theory of Constraints (TOC). It forces the reader to stop looking at individual process efficiencies and start seeing the entire system and its bottlenecks."},
            {"title": "Team of Teams: New Rules of Engagement for a Complex World", "author": "Gen. Stanley McChrystal", "summary": "A modern leadership classic that explains how to transform a rigid command structure into a nimble, adaptive network. Its principles of 'shared consciousness' and 'empowered execution' are the cultural bedrock required for Lean and Kaizen to thrive in the 21st century."},
            {"title": "Thinking, Fast and Slow", "author": "Daniel Kahneman", "summary": "While not a traditional CI book, this Nobel laureate's work is essential for understanding the cognitive biases that affect decision-making. A must-read for leaders who want to make truly data-driven, objective choices."},
        ]
    }

# ==============================================================================
# --- MAIN RENDERING FUNCTION ---
# ==============================================================================

logger = logging.getLogger(__name__)

def render_kaizen_training_hub(ssm: SessionStateManager) -> None:
    """Creates the UI for the Continuous Improvement & Knowledge Hub."""
    st.header("🎓 Continuous Improvement & Knowledge Hub")
    st.markdown("""
    Welcome to the central nervous system of our learning organization. This hub is the catalyst for our Continuous Improvement (CI) culture.
    Here, we **celebrate our successes**, **share our wisdom**, and **empower our teams** with the knowledge to drive process excellence.
    """)

    try:
        # --- HYBRID DATA LOADING STRATEGY ---
        kaizen_events = ssm.get_data("kaizen_events") or get_overhauled_kaizen_data()
        training_materials = ssm.get_data("training_materials") or get_overhauled_training_data()
        glossary = get_glossary_content()
        bibliography = get_bibliography_content()

        # --- UI TABS ---
        events_tab, training_tab, glossary_tab, library_tab = st.tabs([
            "🏆 **Kaizen Event A3 Log**",
            "📚 **Training & Development Library**",
            "📖 **Glossary & Formulas**",
            "🏛️ **Foundational Library**"
        ])

        # ==================== KAIZEN EVENT LOG ====================
        with events_tab:
            st.subheader("A Chronicle of Realized Improvements")
            st.markdown("Each event below is a testament to a team's dedication to making our work better. Review these A3 summaries to understand the 'Why' behind the change and to find inspiration for your own area.")
            df_events = pd.DataFrame(kaizen_events).sort_values(by='date', ascending=False)
            for _, event in df_events.iterrows():
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"#### {event.get('title', 'N/A')}")
                        st.caption(f"**A3 ID:** {event.get('id', 'N/A')} | **Site:** {event.get('site', 'N/A')} | **Completion Date:** {event.get('date', 'N/A')}")
                    with col2:
                        st.button("View Full A3 Report", key=f"report_{event.get('id', 'default_key')}", type="primary", disabled=True, use_container_width=True, help="Full PDF report not available in this demo.")
                    
                    if event.get('problem_background'):
                        st.markdown("**Problem Background:**")
                        st.markdown(f"> {event.get('problem_background')}")
                    
                    if event.get('analysis_and_countermeasures'):
                        with st.expander("**View Detailed Analysis & Countermeasures**"):
                            st.markdown(event.get('analysis_and_countermeasures'))
                            st.caption("_Detailed schematics, raw data, and financial models are redacted from this view and available in the full A3 report._")
                    
                    outcome_key = 'quantified_results' if 'quantified_results' in event else 'outcome'
                    if event.get(outcome_key):
                        st.success(f"**Results:** {event.get(outcome_key)}", icon="💡")

                    if event.get('key_insight'):
                        st.info(f"**Key Insight / Lesson Learned:** {event.get('key_insight')}", icon="🔬")
                st.write("")

        # ==================== TRAINING LIBRARY ====================
        with training_tab:
            st.subheader("Empowering Excellence Through Education")
            st.markdown("A commitment to quality begins with a commitment to learning. This curated library provides resources to develop skills at every level of the organization, from foundational principles to advanced statistical methods.")
            df_training = pd.DataFrame(training_materials)
            for _, material in df_training.iterrows():
                st.markdown(f"""
                <div style="border: 1px solid #c8c8c8; border-left: 6px solid #007bff; border-radius: 8px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="display: flex; align-items: flex-start;">
                        <span style="font-size: 2.5em; margin-right: 25px; margin-top: 5px;">{material.get('icon', '📚')}</span>
                        <div style="flex-grow: 1;">
                            <div style="font-weight: bold; font-size: 1.2em; margin-bottom: 5px;">{material.get('title', 'Untitled Module')}</div>
                            <div style="font-size: 0.9em; color: #555; margin-bottom: 15px;">
                                <span><b>Type:</b> {material.get('type', 'N/A')}</span> | <span><b>Est. Duration:</b> {material.get('duration_hr', '?')} hrs</span> | <span><b>Audience:</b> {material.get('target_audience', 'General')}</span>
                            </div>
                            <p style="font-size: 1em; color: #333; margin-bottom: 15px;">{material.get('description', '')}</p>
                            <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; font-size: 0.9em;">
                                <b>Learning Objectives:</b> <ul>{''.join([f"<li>{obj}</li>" for obj in material.get('learning_objectives', [])])}</ul>
                                <b>Recommended Reading:</b> <i>{material.get('recommended_reading', 'None')}</i>
                            </div>
                            <a href="{material.get('link', '#')}" target="_blank" style="display: inline-block; background-color: #007bff; color: white; padding: 8px 15px; margin-top: 15px; border-radius: 5px; text-decoration: none; font-weight: bold;">Launch Module</a>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # ==================== GLOSSARY & FORMULAS ====================
        with glossary_tab:
            st.subheader("The Common Language of Continuous Improvement")
            st.markdown("Use this dictionary to understand the key terms, concepts, and methodologies that form the foundation of our operational excellence program. A shared vocabulary is essential for effective collaboration and problem-solving.")
            for category, terms in glossary.items():
                with st.expander(f"**{category}**", expanded=(category == "Lean Principles")):
                    for item in terms:
                        st.markdown(f"**{item['term']}**")
                        st.markdown(f"> {item['definition']}")
                        if "formula" in item:
                            st.latex(item["formula"])
                        st.divider()

        # ==================== FOUNDATIONAL LIBRARY ====================
        with library_tab:
            st.subheader("The Foundations of Operational Excellence")
            st.markdown("The following texts are cornerstones of modern quality and leadership thinking. They provide the deep, foundational knowledge that underpins the tools and techniques we use every day. We encourage you to explore these works to further your own journey.")
            for category, books in bibliography.items():
                st.markdown(f"### {category}")
                for book in books:
                    with st.container(border=True):
                        st.markdown(f"**{book['title']}**")
                        st.caption(f"by {book['author']}")
                        st.markdown(book['summary'])
                st.write("")

    except Exception as e:
        st.error(f"An error occurred while rendering the Kaizen & Training Hub: {e}")
        logger.error(f"Failed to render kaizen and training hub: {e}", exc_info=True)
