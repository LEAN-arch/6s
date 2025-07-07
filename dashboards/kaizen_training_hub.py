# 6s/dashboards/kaizen_training_hub.py
"""
Renders the Kaizen & Training Hub.

This module serves as a central repository for documenting continuous improvement
(Kaizen) events and for hosting training materials related to quality principles
and methodologies. It supports the Quality Optimization Engineer's role in
coaching, mentoring, and fostering a culture of quality.
"""

import logging
import pandas as pd
import streamlit as st

from 6s.data.session_state_manager import SessionStateManager

logger = logging.getLogger(__name__)

def render_kaizen_training_hub(ssm: SessionStateManager) -> None:
    """
    Creates the UI for the Kaizen & Training Hub tab.

    Args:
        ssm (SessionStateManager): The session state manager to access kaizen and training data.
    """
    st.header("🎓 Kaizen & Training Hub")
    st.markdown("A central resource for tracking continuous improvement events and accessing training materials on quality methodologies. Use this hub to share best practices and promote a culture of quality at the source across all sites.")

    try:
        # --- 1. Load Data ---
        kaizen_events = ssm.get_data("kaizen_events")
        training_materials = ssm.get_data("training_materials")

        # --- 2. Define Tabs for Events and Materials ---
        st.info("Select a tab to view completed improvement events or to access the training library.", icon="📚")
        events_tab, training_tab = st.tabs(["**Kaizen Event Log**", "**Quality Training Library**"])

        # --- KAIZEN EVENT LOG ---
        with events_tab:
            st.subheader("Completed Continuous Improvement Events")
            st.markdown("A log of completed Kaizen events, documenting the improvements made and the outcomes achieved across different sites. This serves as a knowledge base for sharing successful initiatives.")

            if not kaizen_events:
                st.warning("No Kaizen events have been logged.")
                return

            df_events = pd.DataFrame(kaizen_events)
            df_events['date'] = pd.to_datetime(df_events['date']).dt.strftime('%Y-%m-%d')

            # Display each event as a separate, formatted container
            for index, event in df_events.iterrows():
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"#### {event['title']}")
                        st.caption(f"**ID:** {event['id']} | **Site:** {event['site']} | **Date:** {event['date']}")
                    with col2:
                        # Placeholder for future functionality, e.g., linking to a full report
                        st.button("View Full Report", key=f"report_{event['id']}", disabled=True)

                    st.success(f"**Outcome:** {event['outcome']}", icon="🏆")
                st.write("") # Adds a little vertical space

        # --- TRAINING LIBRARY ---
        with training_tab:
            st.subheader("Quality Methodology Training Library")
            st.markdown("A curated library of training materials covering key quality improvement principles, methodologies, and tools. Use these resources for self-paced learning or as part of formal training sessions.")

            if not training_materials:
                st.warning("No training materials are available.")
                return

            df_training = pd.DataFrame(training_materials)

            # Display materials as a list with icons and links
            for index, material in df_training.iterrows():
                icon_map = {
                    "eLearning": "💻",
                    "PDF Guide": "📄",
                    "Workshop Slides": "📊"
                }
                icon = icon_map.get(material['type'], "🔗")

                st.markdown(f"""
                <div style="display:flex; align-items:center; margin-bottom: 10px;">
                    <span style="font-size: 2em; margin-right: 15px;">{icon}</span>
                    <div>
                        <a href="{material['link']}" style="font-weight: bold; font-size: 1.1em; text-decoration: none; color: #0073e6;">{material['title']}</a><br>
                        <span style="color: #555;">Type: {material['type']} | Estimated Duration: {material['duration_hr']} hours</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error("An error occurred while rendering the Kaizen & Training Hub.")
        logger.error(f"Failed to render kaizen and training hub: {e}", exc_info=True)
