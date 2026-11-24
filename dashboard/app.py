"""Streamlit dashboard for LLM Cost Router analytics."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import httpx

from llm_cost_router.storage import Database

# Page config
st.set_page_config(
    page_title="LLM Cost Router Dashboard",
    page_icon="🎯",
    layout="wide",
)

# Initialize
@st.cache_resource
def get_database():
    """Get database connection."""
    db_path = Path(__file__).parent.parent / "data" / "router.db"
    return Database(db_path)


def format_currency(value):
    """Format currency value."""
    return f"${value:.4f}"


def format_percent(value):
    """Format percentage value."""
    return f"{value:.1f}%"


# Main dashboard
def main():
    st.title("🎯 LLM Cost Router Dashboard")
    st.markdown("Real-time analytics and cost optimization insights")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "📝 Request Audit", "🧪 Live Tester"])

    with tab1:
        show_analytics()

    with tab2:
        show_request_audit()

    with tab3:
        show_live_tester()


def show_analytics():
    """Show main analytics dashboard."""
    db = get_database()
    metrics = db.get_metrics_summary()

    # KPI Cards
    st.subheader("Key Performance Indicators")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Total Requests", f"{metrics['total_requests']:,}")

    with col2:
        st.metric("Actual Spend", format_currency(metrics['total_cost_usd']))

    with col3:
        st.metric("Baseline Spend", format_currency(metrics['baseline_cost_usd']))

    with col4:
        savings_delta = format_currency(metrics['total_savings_usd'])
        st.metric(
            "Total Savings",
            format_currency(metrics['total_savings_usd']),
            delta=f"{metrics['savings_percent']:.1f}%",
            delta_color="normal",
        )

    with col5:
        st.metric("Avg Quality", f"{metrics['avg_quality_score']:.3f}")

    with col6:
        st.metric("Avg Latency", f"{metrics['avg_latency_ms']:.0f}ms")

    st.divider()

    # Charts row 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cumulative Cost: Actual vs Frontier Baseline")
        
        # Get all requests for cumulative calculation
        all_requests = db.get_requests(limit=10000)
        
        if all_requests:
            df_requests = pd.DataFrame(all_requests)
            df_requests['timestamp'] = pd.to_datetime(df_requests['timestamp'])
            df_requests = df_requests.sort_values('timestamp')
            df_requests['cumulative_cost'] = df_requests['cost_usd'].cumsum()
            df_requests['cumulative_baseline'] = df_requests['frontier_baseline_cost_usd'].cumsum()

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df_requests['timestamp'],
                    y=df_requests['cumulative_baseline'],
                    name='Frontier Baseline',
                    line=dict(color='red', dash='dash'),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_requests['timestamp'],
                    y=df_requests['cumulative_cost'],
                    name='Actual Cost',
                    line=dict(color='green'),
                    fill='tonexty',
                )
            )
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Cumulative Cost (USD)",
                hovermode='x unified',
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet. Run some requests to see analytics.")

    with col2:
        st.subheader("Requests by Model & Tier")
        
        tier_dist = metrics.get('tier_distribution', {})
        
        if tier_dist:
            df_tier = pd.DataFrame([
                {"Tier": tier, "Count": count}
                for tier, count in tier_dist.items()
            ])
            
            fig = px.bar(
                df_tier,
                x='Tier',
                y='Count',
                color='Tier',
                color_discrete_map={
                    'local': '#00CC96',
                    'budget': '#636EFA',
                    'balanced': '#EF553B',
                    'frontier': '#AB63FA',
                },
            )
            fig.update_layout(
                xaxis_title="Tier",
                yaxis_title="Request Count",
                showlegend=False,
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Charts row 2
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Complexity Score vs Quality Score")
        
        all_requests = db.get_requests(limit=10000)
        
        if all_requests:
            df_requests = pd.DataFrame(all_requests)
            
            # Join with evaluations
            all_evals = db.get_evaluations(limit=10000)
            if all_evals:
                df_evals = pd.DataFrame(all_evals)
                df_merged = df_requests.merge(
                    df_evals[['request_id', 'quality_score', 'passed']],
                    left_on='id',
                    right_on='request_id',
                    how='left',
                )
                
                df_merged = df_merged.dropna(subset=['quality_score'])
                
                if not df_merged.empty:
                    fig = px.scatter(
                        df_merged,
                        x='complexity_score',
                        y='quality_score',
                        color='tier',
                        size='cost_usd',
                        hover_data=['selected_model', 'cost_usd'],
                        color_discrete_map={
                            'local': '#00CC96',
                            'budget': '#636EFA',
                            'balanced': '#EF553B',
                            'frontier': '#AB63FA',
                        },
                    )
                    fig.update_layout(
                        xaxis_title="Complexity Score",
                        yaxis_title="Quality Score",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Latency by Provider")
        
        all_requests = db.get_requests(limit=10000)
        
        if all_requests:
            df_requests = pd.DataFrame(all_requests)
            
            fig = px.box(
                df_requests,
                x='provider',
                y='latency_ms',
                color='provider',
                points='outliers',
            )
            fig.update_layout(
                xaxis_title="Provider",
                yaxis_title="Latency (ms)",
                showlegend=False,
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Charts row 3
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Quality Pass Rate Over Time")
        
        all_evals = db.get_evaluations(limit=10000)
        
        if all_evals:
            df_evals = pd.DataFrame(all_evals)
            df_evals['timestamp'] = pd.to_datetime(df_evals['timestamp'])
            df_evals = df_evals.sort_values('timestamp')
            
            # Calculate rolling pass rate
            df_evals['passed_int'] = df_evals['passed'].astype(int)
            df_evals['rolling_pass_rate'] = df_evals['passed_int'].rolling(
                window=20, min_periods=1
            ).mean() * 100
            
            fig = px.line(
                df_evals,
                x='timestamp',
                y='rolling_pass_rate',
            )
            fig.add_hline(
                y=75,
                line_dash="dash",
                line_color="red",
                annotation_text="Target (75%)",
            )
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Pass Rate (%)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Adaptive Threshold History")
        
        # Get threshold adjustments
        threshold_adjustments = db.conn.execute(
            "SELECT * FROM threshold_adjustments ORDER BY timestamp DESC LIMIT 50"
        ).fetchall()
        
        if threshold_adjustments:
            df_thresholds = pd.DataFrame([dict(row) for row in threshold_adjustments])
            df_thresholds['timestamp'] = pd.to_datetime(df_thresholds['timestamp'])
            df_thresholds = df_thresholds.sort_values('timestamp')
            
            fig = px.line(
                df_thresholds,
                x='timestamp',
                y='new_threshold',
                color='tier',
                markers=True,
            )
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Threshold Value",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No threshold adjustments yet.")


def show_request_audit():
    """Show detailed request audit log."""
    st.subheader("Request Audit Trail")

    db = get_database()

    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tier_filter = st.selectbox(
            "Filter by Tier",
            ["All", "local", "budget", "balanced", "frontier"],
        )
    
    with col2:
        limit = st.number_input("Rows to show", min_value=10, max_value=500, value=50)
    
    with col3:
        passed_filter = st.selectbox("Quality Filter", ["All", "Passed", "Failed"])

    # Get data
    tier_param = None if tier_filter == "All" else tier_filter
    requests = db.get_requests(limit=limit, tier=tier_param)

    if requests:
        df = pd.DataFrame(requests)
        
        # Join with evaluations
        evals = db.get_evaluations(limit=limit * 2)
        if evals:
            df_evals = pd.DataFrame(evals)
            df = df.merge(
                df_evals[['request_id', 'quality_score', 'passed']],
                left_on='id',
                right_on='request_id',
                how='left',
            )
        
        # Apply passed filter
        if passed_filter == "Passed":
            df = df[df['passed'] == 1]
        elif passed_filter == "Failed":
            df = df[df['passed'] == 0]

        # Format display
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['cost_usd'] = df['cost_usd'].apply(lambda x: f"${x:.6f}")
        df['savings_usd'] = df['savings_usd'].apply(lambda x: f"${x:.6f}")
        df['savings_percent'] = df['savings_percent'].apply(lambda x: f"{x:.1f}%")
        df['quality_score'] = df['quality_score'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

        # Select columns
        display_cols = [
            'timestamp',
            'prompt',
            'complexity_tier',
            'tier',
            'selected_model',
            'cost_usd',
            'savings_usd',
            'savings_percent',
            'quality_score',
            'latency_ms',
        ]
        
        df_display = df[display_cols].copy()
        df_display['prompt'] = df_display['prompt'].str[:100]  # Truncate

        st.dataframe(df_display, use_container_width=True, height=600)

        # Export
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"router_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No requests found.")


def show_live_tester():
    """Interactive prompt tester."""
    st.subheader("Live Prompt Tester")
    st.markdown("Test routing decisions in real-time by calling the router API.")

    # API endpoint
    api_url = st.text_input(
        "Router API URL",
        value="http://localhost:8000/v1/route",
    )

    # Prompt input
    prompt = st.text_area(
        "Enter your prompt:",
        height=150,
        placeholder="e.g., Explain how binary search works with example code",
    )

    # Options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        force_tier = st.selectbox(
            "Force Tier (optional)",
            ["None", "local", "budget", "balanced", "frontier"],
        )
    
    with col2:
        local_only = st.checkbox("Local Only")
    
    with col3:
        run_eval = st.checkbox("Run Evaluation", value=True)

    # Submit
    if st.button("Route Request", type="primary"):
        if not prompt:
            st.error("Please enter a prompt.")
            return

        # Build request
        request_data = {
            "prompt": prompt,
            "local_only": local_only,
            "run_eval": run_eval,
        }
        
        if force_tier != "None":
            request_data["force_tier"] = force_tier

        # Call API
        with st.spinner("Routing request..."):
            try:
                response = httpx.post(
                    api_url,
                    json=request_data,
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

                # Display results
                st.success("Request routed successfully!")

                # Routing decision
                st.subheader("Routing Decision")
                routing = data['routing_decision']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Complexity Tier", routing['complexity_tier'])
                with col2:
                    st.metric("Selected Model", routing['selected_model'])
                with col3:
                    st.metric("Complexity Score", f"{routing['complexity_score']:.3f}")

                st.info(f"**Explanation:** {routing['complexity_explanation']}")

                # Cost metrics
                st.subheader("Cost Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Cost", format_currency(data['cost_usd']))
                with col2:
                    st.metric("Baseline Cost", format_currency(data['frontier_baseline_cost_usd']))
                with col3:
                    st.metric("Savings", format_currency(data['savings_usd']))
                with col4:
                    st.metric("Savings %", format_percent(data['savings_percent']))

                # Response
                st.subheader("Response")
                st.text_area("", value=data['content'], height=200, disabled=True)

                # Evaluation
                if data.get('eval_scores'):
                    st.subheader("Quality Evaluation")
                    eval_data = data['eval_scores']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Quality Score", f"{eval_data['quality_score']:.3f}")
                    with col2:
                        st.metric("Heuristic Score", f"{eval_data['heuristic_score']:.3f}")
                    with col3:
                        judge_score = eval_data.get('judge_score')
                        st.metric(
                            "Judge Score",
                            f"{judge_score:.3f}" if judge_score else "N/A",
                        )
                    
                    if eval_data.get('judge_rationale'):
                        st.info(f"**Judge Rationale:** {eval_data['judge_rationale']}")
                    
                    if eval_data['passed']:
                        st.success("✅ Quality check passed")
                    else:
                        st.warning("⚠️ Quality check failed")

            except httpx.HTTPError as e:
                st.error(f"API request failed: {str(e)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
