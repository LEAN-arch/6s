"""
Renders the Continuous Improvement & Knowledge Hub.

This module serves as a central repository for documenting continuous improvement
(Kaizen) events and for hosting training materials related to quality principles.
It supports the MBB's role in coaching, mentoring, and fostering a culture of
quality by sharing successes and enabling skill development.

SME Overhaul (Kaizen Leader Edition):
- The content has been fully populated with academic-grade, actionable materials
  that reflect the principles of Lean (Imai, Shingo, Ohno) and Six Sigma (Deming).
- The Kaizen Event Log is now structured to mirror A3 Thinking, providing deep
  insights into the problem-solving process, with realistic 'redacted' content.
- The Training Library is now a comprehensive curriculum with detailed descriptions,
  learning objectives, and references to foundational literature.
- The UI/UX has been completely redesigned for a professional, engaging, and
  authoritative "knowledge base" feel, using modern layouts and better visual hierarchy.
- All text has been rewritten to inspire action, promote a learning culture,
  and articulate the 'Why' behind continuous improvement.
"""

import logging
import pandas as pd
import streamlit as st

# NOTE: In a real application, this data would come from the SessionStateManager.
# For this overhaul, we define it here to showcase the rich, academic-grade content.

def get_overhauled_kaizen_data():
    """Generates realistic, detailed Kaizen event data."""
    return [
        {
            "id": "KZN-02",
            "title": "SMED on Stamping Press P-101",
            "site": "Andover, US",
            "date": "2025-06-15",
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
            "id": "KZN-01",
            "title": "5S Implementation in Main Assembly Cell",
            "site": "Eindhoven, NL",
            "date": "2025-05-22",
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
    """Generates a comprehensive, academic-grade training library."""
    return [
        {
            "id": "TRN-101",
            "title": "A3 Thinking: The Art of Problem Solving on a Single Page",
            "type": "eLearning",
            "duration_hr": 2.5,
            "target_audience": "Engineers, Team Leads, Managers",
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
            "target_audience": "Quality Engineers, Process Technicians",
            "link": "#",
            "icon": "📊",
            "description": "A deep dive into the principles of Dr. W. Edwards Deming and Walter Shewhart. This workshop provides the statistical foundation for understanding process variation, distinguishing between common and special causes, and using control charts to monitor and improve process stability.",
            "learning_objectives": [
                "Calculate control limits for I-MR, Xbar-R, and p-charts.",
                "Interpret control chart signals (e.g., Nelson Rules).",
                "Define and calculate process capability indices (Cp, Cpk).",
                "Understand the relationship between process control and process capability."
            ],
            "recommended_reading": "'Understanding Variation: The Key to Managing Chaos' by Donald J. Wheeler"
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

logger = logging.getLogger(__name__)

def render_kaizen_training_hub(ssm: SessionStateManager) -> None:
    """Creates the UI for the Continuous Improvement & Knowledge Hub."""
    st.header("🎓 Continuous Improvement & Knowledge Hub")
    st.markdown("""
    Welcome to the central nervous system of our learning organization. This hub is the catalyst for our Continuous Improvement (CI) culture.
    Here, we **celebrate our successes**, **share our wisdom**, and **empower our teams** with the knowledge to drive process excellence.
    """)

    try:
        # --- 1. Load Data (Using the new, rich content) ---
        kaizen_events = get_overhauled_kaizen_data()
        training_materials = get_overhauled_training_data()

        st.info("Select a tab to review the A3 reports from past Kaizen events or to access our curated library of quality and CI training.", icon="🧠")
        events_tab, training_tab = st.tabs(["🏆 **Kaizen Event A3 Log**", "📚 **Training & Development Library**"])

        # ==================== KAIZEN EVENT LOG ====================
        with events_tab:
            st.subheader("A Chronicle of Realized Improvements")
            st.markdown("Each event below is a testament to a team's dedication to making our work better. Review these A3 summaries to understand the 'Why' behind the change and to find inspiration for your own area.")

            if not kaizen_events:
                st.warning("No Kaizen events have been logged in the data model.")
            else:
                df_events = pd.DataFrame(kaizen_events).sort_values(by='date', ascending=False)
                for _, event in df_events.iterrows():
                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"#### {event['title']}")
                            st.caption(f"**A3 ID:** {event['id']} | **Site:** {event['site']} | **Completion Date:** {event['date']}")
                        with col2:
                            st.button("View Full A3 Report", key=f"report_{event['id']}", type="primary", disabled=True, use_container_width=True)

                        st.markdown("**Problem Background:**")
                        st.markdown(f"> {event['problem_background']}")

                        with st.expander("**View Detailed Analysis & Countermeasures**"):
                            st.markdown(event['analysis_and_countermeasures'])
                            st.caption("_Detailed schematics, raw data, and financial models are redacted from this view and available in the full A3 report._")
                        
                        st.markdown("**Quantified Results:**")
                        st.success(f"{event['quantified_results']}", icon="💡")

                        st.markdown("**Key Insight / Lesson Learned:**")
                        st.info(f"{event['key_insight']}", icon="🔬")

                    st.write("") # Adds vertical space

        # ==================== TRAINING LIBRARY ====================
        with training_tab:
            st.subheader("Empowering Excellence Through Education")
            st.markdown("A commitment to quality begins with a commitment to learning. This curated library provides resources to develop skills at every level of the organization, from foundational principles to advanced statistical methods.")

            if not training_materials:
                st.warning("No training materials are available in the data model.")
            else:
                df_training = pd.DataFrame(training_materials)
                for _, material in df_training.iterrows():
                    st.markdown(f"""
                    <div style="border: 1px solid #c8c8c8; border-left: 6px solid #007bff; border-radius: 8px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <div style="display: flex; align-items: flex-start;">
                            <span style="font-size: 2.5em; margin-right: 25px; margin-top: 5px;">{material['icon']}</span>
                            <div style="flex-grow: 1;">
                                <div style="font-weight: bold; font-size: 1.2em; margin-bottom: 5px;">{material['title']}</div>
                                <div style="font-size: 0.9em; color: #555; margin-bottom: 15px;">
                                    <span><b>Type:</b> {material['type']}</span> |
                                    <span><b>Est. Duration:</b> {material['duration_hr']} hrs</span> |
                                    <span><b>Primary Audience:</b> {material['target_audience']}</span>
                                </div>
                                <p style="font-size: 1em; color: #333; margin-bottom: 15px;">{material['description']}</p>
                                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; font-size: 0.9em;">
                                    <b>Learning Objectives:</b>
                                    <ul>{''.join([f"<li>{obj}</li>" for obj in material['learning_objectives']])}</ul>
                                    <b>Recommended Reading:</b> <i>{material['recommended_reading']}</i>
                                </div>
                                <a href="{material['link']}" target="_blank" style="display: inline-block; background-color: #007bff; color: white; padding: 8px 15px; margin-top: 15px; border-radius: 5px; text-decoration: none; font-weight: bold;">Launch Module</a>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred while rendering the Kaizen & Training Hub: {e}")
        logger.error(f"Failed to render kaizen and training hub: {e}", exc_info=True)
