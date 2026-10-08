t.session_state.window_probabilities = window_probs

                st.success(
                    "Patient analysis completed successfully."
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Analysis failed: {str(e)}"
                )


# ============================================================
# EEG ANALYSIS
# ============================================================

elif page == "🧠 EEG Analysis":

    st.html("""
        <div class="topbar">
            <div class="main-title">
                EEG Analysis
            </div>
            <div class="subtitle">
                19-channel EEG preprocessing and visualization
            </div>
        </div>
        """)

    if st.session_state.eeg_data is None:

        st.info(
            "Analyze a patient first to display EEG results."
        )

    else:

        eeg = st.session_state.eeg_data

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Channels",
                "19"
            )

        with c2:
            st.metric(
                "Sampling Rate",
                "100 Hz"
            )

        with c3:
            st.metric(
                "Window",
                "10 sec"
            )

        channel = st.selectbox(
            "Select EEG Channel",
            EEG_CHANNELS
        )

        channel_index = EEG_CHANNELS.index(
            channel
        )

        st.plotly_chart(
            create_eeg_plot(
                eeg,
                channel_index
            ),
            use_container_width=True
        )

        st.html("""
            <div class="info-pill">
                10-second windows
            </div>

            <div class="info-pill">
                5-second step
            </div>

            <div class="info-pill">
                50% overlap
            </div>

            <div class="info-pill">
                Train-only normalization
            </div>
            """)


# ============================================================
# LIVE MONITORING
# ============================================================

elif page == "📡 Live Monitoring":

    patient_id = st.session_state.get(
        "active_patient_id",
        st.session_state.get(
            "selected_dashboard_patient",
            "0299"
        )
    )

    patient_iot = st.session_state.get(
        "active_patient_iot",
        {}
    ) or {}

    heart_rate = patient_iot.get(
        "heart_rate_mean",
        None
    )

    temperature = patient_iot.get(
        "body_temperature_mean",
        None
    )

    movement = patient_iot.get(
        "movement_mean",
        None
    )

    metrics = [
        (
            "❤️ Heart Rate",
            f"{heart_rate:.2f} BPM"
            if heart_rate is not None
            else "N/A",
            "Stored IoT data"
        ),
        (
            "🌡️ Temperature",
            f"{temperature:.2f} °C"
            if temperature is not None
            else "N/A",
            "Stored IoT data"
        ),
        (
            "📡 Movement",
            f"{movement:.3f}"
            if movement is not None
            else "N/A",
            "Stored IoT data"
        ),
        (
            "🧠 EEG",
            "ACTIVE",
            "19 channels"
        )
    ]

    st.html("""
        <div class="topbar">
            <div class="main-title">
                Live Patient Monitoring
            </div>
            <div class="subtitle">
                Real-time style monitoring interface
                for prototype demonstration
            </div>
        </div>
        """)

    c1, c2, c3, c4 = st.columns(4)

    for col, data in zip(
        [c1, c2, c3, c4],
        metrics
    ):

        with col:

            st.html(
                f"""
                <div class="card">
                    <div class="card-title">
                        {data[0]}
                    </div>
                    <div class="card-value">
                        {data[1]}
                    </div>
                    <div class="card-small">
                        {data[2]}
                    </div>
                </div>
                """
            )

    st.plotly_chart(
        create_monitor_plot(),
        use_container_width=True
    )

    st.html("""
        <div class="warning-box">
        ⚠️ This live-monitoring page is a prototype
        visualization. IoT values shown here are
        demonstration values unless connected to
        the actual sensor system.
        </div>
        """)


# ============================================================
# CLINICAL INFORMATION
# ============================================================

elif page == "🩺 Clinical Information":

    patient_id = st.session_state.get(
        "active_patient_id",
        st.session_state.get(
            "selected_dashboard_patient",
            "0299"
        )
    )

    patient_clinical = st.session_state.get(
        "active_patient_clinical",
        {}
    ) or {}

    age = patient_clinical.get("Age", None)
    sex = patient_clinical.get("Sex", None)
    rosc = patient_clinical.get("ROSC", None)
    ohca = patient_clinical.get("OHCA", None)
    shockable = patient_clinical.get(
        "Shockable_Rhythm",
        None
    )
    ttm = patient_clinical.get("TTM", None)

    def display_value(value, default="N/A"):
        if value is None:
            return default

        try:
            if pd.isna(value):
                return default
        except Exception:
            pass

        return value

    age_display = display_value(age)

    if age_display != "N/A":
        age_display = f"{float(age_display):.0f}"

    sex_display = display_value(sex)

    if sex_display == "N/A":
        sex_encoded = patient_clinical.get(
            "Sex_Encoded",
            None
        )

        if sex_encoded is not None:
            sex_display = (
                "Male"
                if float(sex_encoded) == 0
                else "Female"
            )

    ohca_display = display_value(ohca)

    if ohca_display != "N/A":
        ohca_display = (
            "YES"
            if float(ohca_display) == 1
            else "NO"
        )

    shockable_display = display_value(
        shockable
    )

    if shockable_display != "N/A":
        shockable_display = (
            "YES"
            if float(shockable_display) == 1
            else "NO"
        )

    rosc_display = display_value(rosc)

    if rosc_display != "N/A":
        rosc_display = f"{float(rosc_display):.1f}"

    ttm_display = display_value(ttm)

    if ttm_display != "N/A":
        ttm_display = f"{float(ttm_display):.1f} °C"

    st.html("""
        <div class="topbar">
            <div class="main-title">
                Clinical Information
            </div>
            <div class="subtitle">
                Patient factors used by the multimodal AI model
            </div>
        </div>
        """)

    st.html(
        f"""
        <div class="info-pill">
            Active Patient: {patient_id}
        </div>
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.html(
            f"""
            <div class="card">
                <div class="card-title">Age</div>
                <div class="card-value">
                    {age_display}
                </div>
                <div class="card-small">
                    Patient demographic
                </div>
            </div>
            """
        )

    with c2:

        st.html(
            f"""
            <div class="card">
                <div class="card-title">OHCA</div>
                <div class="card-value">
                    {ohca_display}
                </div>
                <div class="card-small">
                    Out-of-hospital cardiac arrest
                </div>
            </div>
            """
        )

    with c3:

        st.html(
            f"""
            <div class="card">
                <div class="card-title">
                    Shockable Rhythm
                </div>
                <div class="card-value">
                    {shockable_display}
                </div>
                <div class="card-small">
                    Clinical feature
                </div>
            </div>
            """
        )

    st.html(
        '<div class="section-title">'
        'Clinical Features'
        '</div>'
    )

    clinical_table = pd.DataFrame({
        "Feature": [
            "Age",
            "Sex",
            "ROSC",
            "OHCA",
            "Shockable Rhythm",
            "TTM"
        ],
        "Value": [
            f"{age_display} years"
            if age_display != "N/A"
            else "N/A",
            sex_display,
            rosc_display,
            ohca_display,
            shockable_display,
            ttm_display
        ],
        "AI Role": [
            "Demographic",
            "Demographic",
            "Resuscitation",
            "Clinical",
            "Clinical",
            "Temperature management"
        ]
    })

    st.dataframe(
        clinical_table,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# AI PREDICTION
# ============================================================

elif page == "🤖 AI Prediction":

    st.html("""
        <div class="topbar">
            <div class="main-title">
                AI Prediction Center
            </div>
            <div class="subtitle">
                Multimodal Fusion V1 prediction and
                explainable monitoring view
            </div>
        </div>
        """)

    probability = st.session_state.prediction_probability
    status = st.session_state.prediction_status

    left, right = st.columns(
        [1, 1.5]
    )

    with left:

        st.html(
            f"""
            <div class="ai-panel">

                <div class="card-title">
                    PATIENT
                </div>

                <div style="
                    font-size:26px;
                    font-weight:800;
                    margin-top:8px;
                ">
                    {st.session_state.patient_id}
                </div>

                <br>

                <div class="card-title">
                    AI PROBABILITY
                </div>

                <div style="
                    font-size:52px;
                    font-weight:800;
                ">
                    {probability * 100:.2f}%
                </div>

                <div class="
                    {'poor' if status == 'POOR' else 'good'}
                    "
                    style="font-size:20px;">
                    {status}
                </div>

                <br>

                <span class="info-pill">
                    Threshold {DECISION_THRESHOLD:.2f}
                </span>

            </div>
            """
        )

    with right:

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={
                    "suffix": "%"
                },
                title={
                    "text":
                    "Poor-outcome probability"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "threshold": {
                        "line": {
                            "width": 4
                        },
                        "value":
                        DECISION_THRESHOLD * 100
                    }
                }
            )
        )

        fig.update_layout(
            height=320,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if st.session_state.window_probabilities is not None:

        st.html(
            '<div class="section-title">'
            'Window-level Analysis'
            '</div>'
        )

        st.plotly_chart(
            create_probability_plot(
                st.session_state.window_probabilities
            ),
            use_container_width=True
        )

    st.html("""
        <div class="disclaimer">
        <b>Clinical decision-support notice:</b>
        The AI output is an experimental research result
        and must not be treated as an autonomous medical
        diagnosis or treatment decision. Final interpretation
        must be performed by qualified healthcare professionals.
        </div>
        """)

elif page == "🧑‍⚕️ Physiotherapy":

    # ========================================================
    # ACTIVE PATIENT DATA
    # ========================================================
    patient_id = st.session_state.get(
        "active_patient_id",
        st.session_state.get(
            "selected_dashboard_patient",
            "0299"
        )
    )

    prediction_probability = st.session_state.get(
        "prediction_probability",
        None
    )

    prediction_status = st.session_state.get(
        "prediction_status",
        None
    )

    patient_iot = st.session_state.get(
        "active_patient_iot",
        {}
    ) or {}

    patient_clinical = st.session_state.get(
        "active_patient_clinical",
        {}
    ) or {}

    movement_value = patient_iot.get(
        "movement_mean",
        None
    )

    heart_rate = patient_iot.get(
        "heart_rate_mean",
        None
    )

    temperature = patient_iot.get(
        "body_temperature_mean",
        None
    )

    age = patient_clinical.get(
        "Age",
        None
    )

    rosc = patient_clinical.get(
        "ROSC",
        patient_clinical.get("rosc", None)
    )

    ohca = patient_clinical.get(
        "OHCA",
        patient_clinical.get("ohca", None)
    )

    shockable = patient_clinical.get(
        "Shockable Rhythm",
        patient_clinical.get(
            "Shockable_Rhythm",
            patient_clinical.get(
                "shockable_rhythm",
                None
            )
        )
    )

    # ========================================================
    # SAFE DISPLAY HELPER
    # ========================================================

    def safe_value(value, default="N/A"):

        if value is None:
            return default

        try:
            if not np.isfinite(float(value)):
                return default
        except (TypeError, ValueError):
            pass

        return value

    # ========================================================
    # HEADER
    # ========================================================

    st.html("""
        <div class="topbar">
            <div class="main-title">
                Physiotherapy Support
            </div>

            <div class="subtitle">
                Patient-specific AI-assisted rehabilitation
                monitoring for clinician review
            </div>
        """)

    st.html(f"""
        <div style="
            margin-top:12px;
            font-size:16px;
            font-weight:700;
        ">
            Active Patient: {patient_id}
        </div>
    """)

    st.html("<br>")

    # ========================================================
    # PATIENT REHABILITATION CONTEXT
    # ========================================================

    st.html("""
        <div class="section-title">
            AI-assisted rehabilitation support for clinician review
        </div>

        <div class="section-title">
            Patient Rehabilitation Context
        </div>
    """)

    c1, c2, c3, c4 = st.columns(4)

    # --------------------------------------------------------
    # AI PROBABILITY
    # --------------------------------------------------------
    # --------------------------------------------------------
    # MULTIMODAL AI PROBABILITY
    # --------------------------------------------------------

    with c1:

        if prediction_probability is not None:

            try:

                probability_value = float(
                    prediction_probability
                )

                if np.isfinite(probability_value):

                    if probability_value <= 1:

                        probability_text = (
                            f"{probability_value * 100:.2f}%"
                        )

                    else:

                        probability_text = (
                            f"{probability_value:.2f}%"
                        )

                else:

                    probability_text = "N/A"

            except (TypeError, ValueError):

                probability_text = "N/A"

        else:

            probability_text = "N/A"

        st.metric(
            "AI Probability",
            probability_text
        )
    # --------------------------------------------------------
    # AI STATUS
    # --------------------------------------------------------

    with c2:

        if prediction_status in ["GOOD", "POOR"]:

            status_text = prediction_status

        else:

            status_text = "AI Analysis Required"

        st.metric(
            "AI Status",
            status_text
        )

    # --------------------------------------------------------
    # MOVEMENT
    # --------------------------------------------------------

    with c3:

        if movement_value is not None:

            st.metric(
                "Movement",
                f"{movement_value:.2f}"
            )

        else:

            st.metric(
                "Movement",
                "N/A"
            )

    # --------------------------------------------------------
    # HEART RATE
    # --------------------------------------------------------

    with c4:

        if heart_rate is not None:

            st.metric(
                "Heart Rate",
                f"{heart_rate:.1f} BPM"
            )

        else:

            st.metric(
                "Heart Rate",
                "N/A"
            )

    st.html("<br>")

    # ========================================================
    # PATIENT CONTEXT + MOVEMENT MONITORING
    # ========================================================

    p1, p2 = st.columns(2)

    # ========================================================
    # PATIENT CONTEXT BOX
    # ========================================================

    with p1:

        age_display = safe_value(age)

        if age_display != "N/A":

            try:
                age_display = (
                    f"{float(age_display):.0f} years"
                )
            except (TypeError, ValueError):
                pass

        temperature_display = safe_value(
            temperature
        )

        if temperature_display != "N/A":

            try:
                temperature_display = (
                    f"{float(temperature_display):.2f} °C"
                )
            except (TypeError, ValueError):
                pass

        rosc_display = safe_value(rosc)
        ohca_display = safe_value(ohca)
        shockable_display = safe_value(
            shockable
        )

        st.html(f"""
            <div class="card">

                <div class="card-title">
                    PATIENT CONTEXT
                </div>

                <div style="
                    margin-top:14px;
                    line-height:2;
                    font-size:16px;
                ">

                    <b>Age:</b>
                    {age_display}

                    <br>

                    <b>Body Temperature:</b>
                    {temperature_display}

                    <br>

                    <b>ROSC:</b>
                    {rosc_display}

                    <br>

                    <b>OHCA:</b>
                    {ohca_display}

                    <br>

                    <b>Shockable Rhythm:</b>
                    {shockable_display}

                </div>

            </div>
        """)

    # ========================================================
    # MOVEMENT MONITORING BOX
    # ========================================================

    with p2:

        if movement_value is not None:

            movement_display = (
                f"{movement_value:.3f}"
            )

            if movement_value < 0.30:

                movement_category = (
                    "Low Mobility"
                )

                movement_note = (
                    "Low movement level observed "
                    "in the stored monitoring data."
                )

            elif movement_value < 0.70:

                movement_category = (
                    "Moderate Mobility"
                )

                movement_note = (
                    "Moderate movement level observed "
                    "in the stored monitoring data."
                )

            else:

                movement_category = (
                    "Higher Mobility"
                )

                movement_note = (
                    "Higher movement level observed "
                    "in the stored monitoring data."
                )

        else:

            movement_display = "N/A"

            movement_category = (
                "Unavailable"
            )

            movement_note = (
                "Movement data is not currently available."
            )

        st.html(f"""
            <div class="card">

                <div class="card-title">
                    MOVEMENT MONITORING
                </div>

                <div style="
                    margin-top:14px;
                    font-size:22px;
                    font-weight:800;
                ">
                    {movement_display}
                </div>

                <div style="
                    margin-top:6px;
                    font-size:15px;
                    font-weight:700;
                ">
                    {movement_category}
                </div>

                <div style="
                    margin-top:12px;
                    line-height:1.6;
                    font-size:15px;
                ">
                    {movement_note}
                </div>

            </div>
        """)

    st.html("<br>")

    # ========================================================
    # AI VIDEO RECOMMENDATION
    # ========================================================

    st.html("""
        <div class="section-title">
            🤖 AI-Assisted Rehabilitation Video Recommendation
        </div>
    """)

    # ========================================================
    # RECOMMENDATION LOGIC
    # ========================================================

    if movement_value is not None:

        if movement_value < 0.30:

            recommendation_type = (
                "Low-Mobility Rehabilitation Support"
            )

            recommendation_reason = (
                f"Movement mean is {movement_value:.3f}. "
                "The system has selected a low-mobility "
                "rehabilitation education category for "
                "clinician review."
            )

            video_title = (
                "Stroke Rehabilitation – "
                "Mobility and Movement Exercises"
            )

            video_url = (
                "https://www.youtube.com/results"
                "?search_query=stroke+rehabilitation"
                "+mobility+exercises+physiotherapy"
            )

        elif movement_value < 0.70:

            recommendation_type = (
                "Supported Mobility Rehabilitation"
            )

            recommendation_reason = (
                f"Movement mean is {movement_value:.3f}. "
                "The system has selected a supported mobility "
                "rehabilitation education category for "
                "clinician review."
            )

            video_title = (
                "Stroke Rehabilitation – "
                "Standing and Mobility Exercises"
            )

            video_url = (
                "https://www.youtube.com/results"
                "?search_query=stroke+rehabilitation"
                "+standing+mobility+exercises"
            )

        else:

            recommendation_type = (
                "Active Mobility Rehabilitation Support"
            )

            recommendation_reason = (
                f"Movement mean is {movement_value:.3f}. "
                "The system has selected an active mobility "
                "rehabilitation education category for "
                "clinician review."
            )

            video_title = (
                "Stroke Rehabilitation – "
                "Balance and Mobility Exercises"
            )

            video_url = (
                "https://www.youtube.com/results"
                "?search_query=stroke+rehabilitation"
                "+balance+mobility+exercises"
            )

    else:

        recommendation_type = (
            "General Rehabilitation Education"
        )

        recommendation_reason = (
            "Movement data is unavailable. "
            "A movement-specific recommendation "
            "cannot be generated."
        )

        video_title = (
            "Stroke Rehabilitation – "
            "General Physiotherapy Education"
        )

        video_url = (
            "https://www.youtube.com/results"
            "?search_query=stroke+rehabilitation"
            "+physiotherapy+education"
        )

    # ========================================================
    # RECOMMENDATION CARD
    # ========================================================

    st.html(f"""
        <div class="card">

            <div class="card-title">
                🎥 AI VIDEO RECOMMENDATION
            </div>

            <div style="
                margin-top:14px;
                font-size:22px;
                font-weight:800;
            ">
                {recommendation_type}
            </div>

            <div style="
                margin-top:14px;
                font-size:15px;
                line-height:1.6;
            ">

                <b>Why this was selected:</b><br>

                {recommendation_reason}

            </div>

            <div style="
                margin-top:18px;
                font-size:17px;
                font-weight:700;
            ">
                Recommended Educational Video
            </div>

            <div style="
                margin-top:8px;
                font-size:16px;
            ">
                🎥 {video_title}
            </div>

        </div>
    """)

    st.html("<br>")

    st.link_button(
        "▶ Watch Recommended Rehabilitation Video",
        video_url,
        width="stretch"
    )

    st.html("<br>")

    # ========================================================
    # CLINICIAN REVIEW PROTOCOL
    # ========================================================

    st.html("""
        <div class="section-title">
            Clinician Review Protocol
        </div>
    """)

    r1, r2 = st.columns(2)

    with r1:

        st.html("""
            <div class="card">

                <div class="card-title">
                    POSITIONING
                </div>

                <div style="
                    margin-top:10px;
                    font-size:19px;
                    font-weight:800;
                ">
                    Pressure-relief assessment
                </div>

                <div style="
                    margin-top:10px;
                    line-height:1.6;
                    font-size:15px;
                ">
                    Review patient positioning, pressure
                    points and bed positioning according
                    to the responsible clinical protocol.
                </div>

            </div>
        """)

        st.html("<br>")

        st.html("""
            <div class="card">

                <div class="card-title">
                    PASSIVE MOVEMENT
                </div>

                <div style="
                    margin-top:10px;
                    font-size:19px;
                    font-weight:800;
                ">
                    Range-of-motion assessment
                </div>

                <div style="
                    margin-top:10px;
                    line-height:1.6;
                    font-size:15px;
                ">
                    Clinician may assess passive range of
                    motion and determine whether appropriate
                    rehabilitation activity is indicated.
                </div>

            </div>
        """)

    with r2:

        st.html("""
            <div class="card">

                <div class="card-title">
                    MOVEMENT RESPONSE
                </div>

                <div style="
                    margin-top:10px;
                    font-size:19px;
                    font-weight:800;
                ">
                    Monitor movement behaviour
                </div>

                <div style="
                    margin-top:10px;
                    line-height:1.6;
                    font-size:15px;
                ">
                    Compare movement measurements across
                    monitoring windows and review changes
                    with the patient's clinical condition.
                </div>

            </div>
        """)

        st.html("<br>")

        if prediction_status == "POOR":

            escalation_title = (
                "Enhanced clinician review"
            )

            escalation_text = (
                "The current AI output is classified as "
                "POOR according to the project decision "
                "threshold. The responsible clinical team "
                "should review the patient before making "
                "rehabilitation decisions."
            )

        elif prediction_status == "GOOD":

            escalation_title = (
                "Routine clinician review"
            )

            escalation_text = (
                "The current AI output is classified as "
                "GOOD according to the project decision "
                "threshold. Continue clinical assessment "
                "and monitoring before making rehabilitation "
                "decisions."
            )

        else:

            escalation_title = (
                "AI assessment required"
            )

            escalation_text = (
                "Run AI analysis for the active patient "
                "to display the current model output. "
                "Clinical assessment remains required."
            )

        st.html(f"""
            <div class="card">

                <div class="card-title">
                    CLINICAL ESCALATION
                </div>

                <div style="
                    margin-top:10px;
                    font-size:19px;
                    font-weight:800;
                ">
                    {escalation_title}
                </div>

                <div style="
                    margin-top:10px;
                    line-height:1.6;
                    font-size:15px;
                ">
                    {escalation_text}
                </div>

            </div>
        """)

    st.html("<br>")

    # ========================================================
    # SINGLE FINAL DISCLAIMER
    # ========================================================

    st.html("""
        <div style="
            margin-top:20px;
            padding:16px;
            border-radius:10px;
            font-size:14px;
            line-height:1.6;
        ">

            🧑‍⚕️ <b>Clinician Review Required</b>

            <br><br>

            The Smart Coma Patient Monitoring System provides
            AI-assisted rehabilitation-support information
            and educational video recommendations.

            <br><br>

            It does not autonomously diagnose, prescribe,
            or order treatment.

        </div>
    """)   
# ============================================================
# ABOUT
# ============================================================

elif page == "🔬 About Project":

    st.html("""
        <div class="topbar">
            <div class="main-title">
                ARMedisense / SmartComa
            </div>
            <div class="subtitle">
                AI-based multimodal monitoring system
                for intelligent coma-patient monitoring
            </div>
        </div>
        """)

    st.html("""
        <div class="ai-panel">

        <h3>System Architecture</h3>

        <div class="pipeline">

            <div class="pipeline-item">
                EEG Dataset
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                EEG Preprocessing
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                CNN + BiLSTM
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                IoT Features
            </div>

            <div class="pipeline-arrow">+</div>

            <div class="pipeline-item">
                Clinical Features
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                Fusion V1
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                Patient-level AI Result
            </div>

        </div>

        </div>
        """)

    st.html(
        '<div class="section-title">'
        'Technologies'
        '</div>'
    )

    tech = pd.DataFrame({
        "Layer": [
            "EEG",
            "Machine Learning",
            "IoT",
            "Dashboard",
            "AR"
        ],
        "Technology": [
            "PhysioNet I-CARE EEG",
            "TensorFlow / Keras",
            "Heart Rate / Temperature / Movement",
            "Python / Streamlit / Plotly",
            "Unity / ARCore"
        ]
    })

    st.dataframe(
        tech,
        use_container_width=True,
        hide_index=True
    )

    st.html("""
        <div class="disclaimer">

        <b>Research prototype:</b>
        This system is intended for academic demonstration
        and AI-assisted clinical decision support.
        It is not a certified medical device and does not
        replace professional diagnosis, monitoring or treatment.

        </div>
        """)


# ============================================================
# FOOTER
# ============================================================

st.html("""
    <div class="footer">
        SMARTCOMA AI • Multimodal Patient Monitoring System
        <br>
        EEG + IoT + Clinical Intelligence • Academic Research Prototype
    </div>
    """)
