"""
Renders the Continuous Improvement & Knowledge Hub.

This module serves as a central repository for documenting continuous improvement
(Kaizen) events and for hosting training materials related to quality principles.
It supports the MBB's role in coaching, mentoring, and fostering a culture of
quality by sharing successes and enabling skill development.

SME Overhaul:
- The UI/UX has been completely redesigned for a more engaging and professional
  "knowledge base" feel.
- The Kaizen Event Log now uses distinct, bordered containers for each event,
  improving readability and highlighting individual successes.
- The Training Library now uses a modern, icon-driven "card" layout, which is
  more visually appealing and easier to navigate than a simple list.
- All text and titles have been refined to better articulate the hub's purpose
  of celebrating wins and enabling growth.
- Placeholder UI elements (like a disabled "View Report" button) have been
  added to suggest future scalability.
"""

import logging
import pandas as pd
import streamlit as st

from six_sigma.data.session_state_manager import SessionStateManager

logger = logging.getLogger(__name__)

def render_kaizen_training_hub(ssm: SessionStateManager) -> None:
    """Creates the UI for the Continuous Improvement & Knowledge Hub."""
    st.header("🎓 Continuous Improvement & Knowledge Hub")
    st.markdown("A central resource for celebrating wins, sharing best practices, and promoting a culture of quality. Use this hub to review completed improvement events and to access the training library.")

    try:
        # --- 1. Load Data ---
        kaizen_events = ssm.get_data("kaizen_events")
        training_materials = ssm.get_data("training_materials")

        # --- 2. Define Tabs for Events and Materials ---
        st.info("Select a tab to view the log of completed improvement events or to access the quality training library.", icon="🧠")
        events_tab, training_tab = st.tabs(["🏆 **Kaizen Event Log**", "📚 **Training Library**"])

        # ==================== KAIZEN EVENT LOG ====================
        with events_tab:
            st.subheader("Completed Continuous Improvement Events")
            st.markdown("A log of completed Kaizen events, documenting the improvements made and the outcomes achieved across different sites. This serves as a knowledge base for sharing successful initiatives.")

            if not kaizen_events:
                st.warning("No Kaizen events have been logged in the data model.")
            else:
                df_events = pd.DataFrame(kaizen_events)
                # Ensure date is in a consistent, readable format
                df_events['date'] = pd.to_datetime(df_events['date']).dt.strftime('%Y-%m-%d')
                df_events = df_events.sort_values(by='date', ascending=False)

                # Display each event as a separate, formatted container
                for _, event in df_events.iterrows():
                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"#### {event['title']}")
                            st.caption(f"**ID:** {event['id']} | **Site:** {event['site']} | **Date:** {event['date']}")
                        with col2:
                            # Placeholder for future functionality, e.g., linking to a full report
                            st.button("View Full Report", key=f"report_{event['id']}", disabled=True, use_container_width=True)

                        st.success(f"**Outcome:** {event['outcome']}", icon="💡")
                    st.write("") # Adds a little vertical space between events

        # ==================== TRAINING LIBRARY ====================
        with training_tab:
            st.subheader("Quality Methodology Training Library")
            st.markdown("A curated library of training materials covering key quality improvement principles, methodologies, and tools. Use these resources for self-paced learning or as part of formal training sessions.")

            if not training_materials:
                st.warning("No training materials are available in the data model.")
            else:
                df_training = pd.DataFrame(training_materials)

                # Display materials as a list of styled "cards"
                for _, material in df_training.iterrows():
                    icon_map = {
                        "eLearning": "💻",
                        "PDF Guide": "📄",
                        "Workshop Slides": "📊",
                        "Video": "🎥"
                    }
                    icon = icon_map.get(material['type'], "🔗")

                    st.markdown(f"""
                    <div style="border: 1px solid #e0e0e0; border-radius: 5px; padding: 15px; margin-bottom: 10px; display: flex; align-items: center;">
                        <span style="font-size: 2.5em; margin-right: 20px;">{icon}</span>
                        <div>
                            <a href="{material['link']}" style="font-weight: bold; font-size: 1.1em; text-decoration: none; color: #007bff;" target="_blank">{material['title']}</a>
                            <div style="font-size: 0.9em; color: #6c757d;">
                                <span>Type: {material['type']}</span> |
                                <span>Duration: {material['duration_hr']} hrs</span> |
                                <span>Audience: {material['target_audience']}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred while rendering the Kaizen & Training Hub: {e}")
        logger.error(f"Failed to render kaizen and training hub: {e}", exc_info=True)
