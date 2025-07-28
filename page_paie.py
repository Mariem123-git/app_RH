
from utils import recherche_employe
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from uploader import uploader_excel




def run(df, xls, uploaded_file):
    import streamlit as st

    df_filtered = recherche_employe(df)

    # Option : utiliser le DataFrame filtré ou non
    use_filtered = st.checkbox("📊 Utiliser le filtre pour les graphiques", value=False)
    if use_filtered:
        df = df_filtered

    # CSS personnalisé pour styliser la section paie
    st.markdown("""
        <style>
            /* Cartes métriques pour la paie */
            .paie-metric-card {
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                text-align: center;
                margin: 10px;
                border: 1px solid rgba(255,255,255,0.2);
                backdrop-filter: blur(10px);
            }

            .paie-metric-card h3 {
                color: white;
                margin-bottom: 10px;
                font-size: 1.1em;
                font-weight: bold;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
            }

            .paie-metric-card p {
                color: #f0f0f0;
                font-size: 1.8em;
                margin: 0;
                font-weight: bold;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
            }

            /* Cartes spécifiques pour la paie */
            .card-hs {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }

            .card-rendement {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            }

            .card-anciennete-prime {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            }

            .card-poste {
                background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            }

            .card-total-primes {
                background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            }

            /* Section header pour la paie */
            .paie-section-header {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 15px;
                margin: 25px 0;
                text-align: center;
                box-shadow: 0 6px 20px rgba(0,0,0,0.2);
                border: 2px solid rgba(255,255,255,0.1);
            }

            .paie-section-header h2 {
                margin: 0;
                font-size: 1.8em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }

            /* Cartes pour top 5 employés */
            .top-employee-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 15px;
                border-radius: 10px;
                margin: 8px 0;
                color: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .top-employee-card .employee-name {
                font-weight: bold;
                font-size: 1.1em;
            }

            .top-employee-card .employee-prime {
                font-size: 1.2em;
                color: #fee140;
                font-weight: bold;
            }

            /* Graphiques avec fond coloré */
            .chart-container {
                background: rgba(255,255,255,0.9);
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 6px 20px rgba(0,0,0,0.1);
                margin: 20px 0;
            }

            /* Alertes stylisées */
            .custom-warning {
                background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
                color: #2d3436;
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid #e17055;
                margin: 15px 0;
                font-weight: bold;
            }

            .custom-info {
                background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
                color: white;
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid #0984e3;
                margin: 15px 0;
                font-weight: bold;
            }

            .custom-error {
                background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
                color: white;
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid #e84393;
                margin: 15px 0;
                font-weight: bold;
            }
        </style>
        """, unsafe_allow_html=True)

    import streamlit as st
    def process_paie_data(df, uploaded_file=None):
        absences_Section = None

        # Header principal
        st.markdown('<div class="paie-section-header"><h2> Statistiques Paie</h2></div>', unsafe_allow_html=True)

        # Nettoyer : supprimer la dernière ligne souvent vide
        df = df.iloc[:-1, :]

        # Convertir les colonnes à valeurs numériques (sécurisé)
        primes_cols = ['Salaire des HS', 'Prime de rendement', "Prime d'ancienneté", 'Prime de poste']
        for col in primes_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        def safe_sum(df, col):
            if col in df.columns and df[col].notna().any():
                return pd.to_numeric(df[col], errors='coerce').sum()
            else:
                return 0

        def safe_mean(df, col):
            if col in df.columns and df[col].notna().any():
                return pd.to_numeric(df[col], errors='coerce').mean()
            else:
                return 0

        # Calculs principaux
        total_HS = safe_sum(df, 'Salaire des HS')
        prime_rendement_moy = safe_mean(df, 'Prime de rendement')
        prime_anciennete_moy = safe_mean(df, "Prime d'ancienneté")
        prime_poste_moy = safe_mean(df, 'Prime de poste')
        prime_salisseur_moy = safe_mean(df, 'Prime de Salissure')

        total_primes = (
                safe_sum(df, 'Prime de rendement') +
                safe_sum(df, "Prime d'ancienneté") +
                safe_sum(df, 'Prime de poste') +
                safe_sum(df, 'Prime de Salissure')
        )

        if 'Heures suppl' in df.columns:
            total_nb_HS = df['Heures suppl'].sum()
        else:
            total_nb_HS = 0

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
                            <div class="paie-metric-card card-rendement">
                                <h3> Nombre total des heures supplémentaires</h3>
                                <p>{total_nb_HS}</p>
                            </div>
                            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class="paie-metric-card card-hs">
                    <h3> Montant Total des heures supplémentaires</h3>
                    <p>{total_HS:,.2f} DH</p>
                </div>
                """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div class="paie-metric-card card-rendement">
                    <h3> Prime de rendement moyen</h3>
                    <p>{prime_rendement_moy:,.2f} DH</p>
                </div>
                """, unsafe_allow_html=True)

        col4, col5, col6, col7 = st.columns(4)

        with col4:
            st.markdown(f"""
                <div class="paie-metric-card card-anciennete-prime">
                    <h3>Prime d'ancienneté moyenne</h3>
                    <p>{prime_anciennete_moy:,.2f} DH</p>
                </div>
                """, unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
                <div class="paie-metric-card card-poste">
                    <h3> Prime de poste moyenne</h3>
                    <p>{prime_poste_moy:,.2f} DH</p>
                </div>
                """, unsafe_allow_html=True)

        with col6:
            st.markdown(f"""
                <div class="paie-metric-card card-total-primes">
                    <h3> Total des primes moyens </h3>
                    <p>{total_primes:,.2f} DH</p>
                </div>
                """, unsafe_allow_html=True)
        with col7:
            st.markdown(f"""
                           <div class="paie-metric-card card-rendement">
                               <h3> Prime de Salissure moyen</h3>
                               <p>{prime_salisseur_moy:,.2f} DH</p>
                           </div>
                           """, unsafe_allow_html=True)

        # Fonction pour graphique camembert des primes + Top 5
        def plot_pie_primes_styled(df):
            primes = ['Prime de rendement', "Prime d'ancienneté", 'Prime de poste', 'Prime de Salissure']
            primes = [col for col in primes if col in df.columns]
            if 'Noms & Prénoms' in df.columns and primes:
                df['Primes totales'] = df[primes].sum(axis=1)
                top_5 = df[['Noms & Prénoms', 'Primes totales']].sort_values(by='Primes totales', ascending=False).head(
                    5)
            else:
                top_5 = pd.DataFrame(columns=['Noms & Prénoms', 'Primes totales'])

            if not primes:
                st.markdown('<div class="custom-warning">⚠️ Aucune colonne de primes trouvée pour le graphique.</div>',
                            unsafe_allow_html=True)
                return

            moyennes, labels = [], []
            for col in primes:
                moyenne = pd.to_numeric(df[col], errors='coerce').dropna()
                if not moyenne.empty:
                    moyennes.append(moyenne.mean())
                    labels.append(col.replace('Prime de ', '').replace("Prime d'", '').title())

            if not moyennes:
                st.markdown('<div class="custom-warning">⚠️ Aucune donnée valide pour les primes.</div>',
                            unsafe_allow_html=True)
                return

            # Créer deux colonnes pour l'affichage
            col1, col2 = st.columns([1, 1])

            with col1:
                # Top 5 employés avec le plus de primes

                st.markdown('<div class="paie-section-header"><h2> Top 5 Employés - Primes</h2></div>',
                            unsafe_allow_html=True)

                # Affichage stylisé du top 5
                for i, (_, row) in enumerate(top_5.iterrows()):
                    medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i + 1}."
                    st.markdown(f"""
                            <div class="top-employee-card">
                                <div>
                                    <span style="font-size: 1.2em; margin-right: 10px;">{medal}</span>
                                    <span class="employee-name">{row['Noms & Prénoms']}</span>
                                </div>
                                <div class="employee-prime">{row['Primes totales']:,.2f} DH</div>
                            </div>
                            """, unsafe_allow_html=True)

            with col2:
                # Graphique camembert stylisé
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)

                fig_pr, ax = plt.subplots(figsize=(8, 8))
                fig_pr.patch.set_facecolor('#f0f2f6')

                # Couleurs personnalisées
                colors = ['#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a'][:len(moyennes)]

                # Créer le camembert avec des effets visuels
                wedges, texts, autotexts = ax.pie(
                    moyennes,
                    labels=labels,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors,
                    shadow=True,
                    explode=[0.05] * len(moyennes),
                    textprops={'fontsize': 12, 'fontweight': 'bold'}
                )

                # Améliorer l'apparence du texte
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

                ax.set_title('Répartition des Primes Moyennes',
                             fontsize=16, fontweight='bold', pad=20)

                plt.tight_layout()
                st.pyplot(fig_pr)
                return fig_pr

                st.markdown('</div>', unsafe_allow_html=True)

        # Appel de la fonction
        fig_pr = plot_pie_primes_styled(df)


        # Fusion Paie & Informations Générales
        if uploaded_file:
            try:
                xls = pd.ExcelFile(uploaded_file)
                df_paie = xls.parse("Paie – Données de base")
                df_infos = xls.parse("Informations Générales")

                if 'Noms & Prénoms' in df_paie.columns and 'Noms & Prénoms' in df_infos.columns:
                    df_merge = pd.merge(df_paie, df_infos[['Noms & Prénoms', 'Section']], on='Noms & Prénoms',
                                        how='left')

                    # Heures supplémentaires par Section
                    if 'Heures suppl' in df_merge.columns and 'Section' in df_merge.columns:
                        hs_par_Section = df_merge.groupby('Section')['Heures suppl'].sum().sort_values(ascending=False)

                        st.markdown(
                            '<div class="paie-section-header"><h2>📊 Heures Supplémentaires par Section</h2></div>',
                            unsafe_allow_html=True)

                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

                        fig_hs, ax = plt.subplots(figsize=(12, 6))
                        fig_hs.patch.set_facecolor('#f0f2f6')
                        ax.set_facecolor('#ffffff')

                        # Dégradé de couleurs pour les barres
                        colors = plt.cm.viridis(np.linspace(0, 1, len(hs_par_Section)))

                        bars = ax.bar(hs_par_Section.index, hs_par_Section.values, color=colors,
                                      edgecolor='white', linewidth=2)

                        # Ajouter les valeurs sur les barres
                        for bar, value in zip(bars, hs_par_Section.values):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width() / 2., height + max(hs_par_Section.values) * 0.01,
                                    f'{value:.0f}h', ha='center', va='bottom', fontweight='bold')

                        ax.set_xlabel('Section', fontweight='bold', fontsize=12)
                        ax.set_ylabel('Heures Supplémentaires', fontweight='bold', fontsize=12)
                        ax.set_title('Distribution des Heures Supplémentaires par Section',
                                     fontweight='bold', fontsize=14, pad=20)

                        plt.xticks(rotation=45, ha='right')
                        ax.grid(axis='y', alpha=0.3, linestyle='--')
                        ax.set_axisbelow(True)

                        plt.tight_layout()
                        st.pyplot(fig_hs)

                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="custom-warning">⚠️ Clé \'Noms & Prénoms\' manquante pour la jointure.</div>',
                        unsafe_allow_html=True)

                # Taux d'absentéisme moyen par Section
                if all(col in df_paie.columns for col in
                       ['Jours d\'absences', 'Nbre Jours']) and 'Section' in df_infos.columns:
                    if 'Noms & Prénoms' in df_paie.columns and 'Noms & Prénoms' in df_infos.columns:
                        df_paie['Noms & Prénoms'] = df_paie['Noms & Prénoms'].astype(str).str.strip().str.lower()
                        df_infos['Noms & Prénoms'] = df_infos['Noms & Prénoms'].astype(str).str.strip().str.lower()

                        df_merged = pd.merge(df_paie, df_infos[['Noms & Prénoms', 'Section']], on='Noms & Prénoms',
                                             how='left')
                        fig_abs = None
                        # Convertir en numérique
                        df_merged['Nbre Jours'] = pd.to_numeric(df_merged['Nbre Jours'], errors='coerce')
                        df_merged['Jours d\'absences'] = pd.to_numeric(df_merged['Jours d\'absences'], errors='coerce')

                        # Calculer la colonne dans df_merged globalement
                        df_merged['Taux d\'absence (%)'] = (df_merged['Jours d\'absences'] / df_merged[
                            'Nbre Jours']) * 100

                        # Puis filtrer pour garder seulement les lignes valides
                        df_merged_filtered = df_merged[
                            (df_merged['Nbre Jours'] > 0) &
                            (df_merged['Jours d\'absences'].notna()) &
                            (df_merged['Jours d\'absences'] > 0)
                            ]

                        if not df_merged_filtered.empty:

                            absences_Section = df_merged_filtered.groupby('Section')[
                                'Taux d\'absence (%)'].mean().sort_values(ascending=False)


                            absences_Section = df_merged.groupby('Section')['Taux d\'absence (%)'].mean().sort_values(
                                ascending=False)

                            st.markdown(
                                '<div class="paie-section-header"><h2>📊 Taux d\'Absentéisme par Section</h2></div>',
                                unsafe_allow_html=True)

                            # Métriques d'absentéisme
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                taux_global = df_merged['Taux d\'absence (%)'].mean()
                                st.markdown(f"""
                                    <div class="paie-metric-card" style="background: linear-gradient(135deg, #ff7675 0%, #d63031 100%);">
                                        <h3> Taux Global</h3>
                                        <p>{taux_global:.1f}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                            with col2:
                                section_max = absences_Section.idxmax()
                                taux_max = absences_Section.max()
                                st.markdown(f"""
                                    <div class="paie-metric-card" style="background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);">
                                        <h3> Section Critique</h3>
                                        <p>{section_max}</p>
                                        <small style="color: #fff; font-size: 0.9em;">{taux_max:.1f}%</small>
                                    </div>
                                    """, unsafe_allow_html=True)

                            with col3:
                                section_min = absences_Section.idxmin()
                                taux_min = absences_Section.min()
                                st.markdown(f"""
                                    <div class="paie-metric-card" style="background: linear-gradient(135deg, #00b894 0%, #00a085 100%);">
                                        <h3> Meilleure Section</h3>
                                        <p>{section_min}</p>
                                        <small style="color: #fff; font-size: 0.9em;">{taux_min:.1f}%</small>
                                    </div>
                                    """, unsafe_allow_html=True)

                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

                            fig_abs, ax = plt.subplots(figsize=(12, 6))
                            fig_abs.patch.set_facecolor('#f0f2f6')
                            ax.set_facecolor('#ffffff')

                            # Couleurs basées sur le taux (rouge pour élevé, vert pour faible)
                            colors = []
                            for taux in absences_Section.values:
                                if taux > 10:
                                    colors.append('#ff7675')  # Rouge
                                elif taux > 5:
                                    colors.append('#fdcb6e')  # Orange
                                else:
                                    colors.append('#00b894')  # Vert

                            bars = ax.bar(absences_Section.index, absences_Section.values, color=colors,
                                          edgecolor='white', linewidth=2)

                            # Ajouter les valeurs sur les barres
                            for bar, value in zip(bars, absences_Section.values):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width() / 2.,
                                        height + max(absences_Section.values) * 0.01,
                                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

                            ax.set_xlabel('Section', fontweight='bold', fontsize=12)
                            ax.set_ylabel('Taux d\'Absence (%)', fontweight='bold', fontsize=12)
                            ax.set_title('Taux d\'Absentéisme Moyen par Section',
                                         fontweight='bold', fontsize=14, pad=20)

                            plt.xticks(rotation=45, ha='right')
                            ax.grid(axis='y', alpha=0.3, linestyle='--')
                            ax.set_axisbelow(True)

                            plt.tight_layout()
                            st.pyplot(fig_abs)

                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            absences_Section = None
                            st.markdown(
                                '<div class="custom-info">ℹ️ La colonne \'Jours d\'absences\' est vide ou n\'a que des zéros.</div>',
                                unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<div class="custom-error">❌ Colonne \'Noms & Prénoms\' manquante pour fusionner.</div>',
                            unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="custom-warning">⚠️ Colonnes manquantes pour calculer le taux d\'absentéisme.</div>',
                        unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f'<div class="custom-error">❌ Erreur lors du traitement du fichier: {str(e)}</div>',
                            unsafe_allow_html=True)

        import io

        primes = ['Prime de rendement', "Prime d'ancienneté", 'Prime de poste', 'Prime de Salissure']
        primes = [col for col in primes if col in df.columns]

        if 'Noms & Prénoms' in df.columns and primes:
            df['Primes totales'] = df[primes].sum(axis=1)
            top_5 = df[['Noms & Prénoms', 'Primes totales']].sort_values(
                by='Primes totales', ascending=False).head(5)
        else:
            top_5 = pd.DataFrame(columns=['Noms & Prénoms', 'Primes totales'])

        # === 1️⃣ DataFrame Statistiques ===
        df_stats = pd.DataFrame({
            'Indicateur': [
                'Nombre total des heures supplémentaires',
                'Salaire total des HS',
                'Prime de rendement moyenne',
                'Prime d\'ancienneté moyenne',
                'Prime de poste moyenne',
                'Prime de salissure moyenne',
                'Total des primes',
                ''

            ],
            'Valeur': [
                total_nb_HS,
                total_HS,
                prime_rendement_moy,
                prime_anciennete_moy,
                prime_poste_moy,
                prime_salisseur_moy,
                total_primes,
                ''
            ]
        })

        # === Calculs ===
        if absences_Section is not None and not absences_Section.empty:
            taux_global = df_merged['Taux d\'absence (%)'].mean()
            section_max = absences_Section.idxmax()
            taux_max = absences_Section.max()
            section_min = absences_Section.idxmin()
            taux_min = absences_Section.min()
        else:
            taux_global = 0
            section_max = ''
            taux_max = 0
            section_min = ''
            taux_min = 0

        # === Ajouter aux statistiques ===
        df_stats_abs = pd.DataFrame({
            'Indicateur': [
                'Taux d\'absentéisme global',
                'Section avec taux max',
                'Taux d\'absentéisme max',
                'Section avec taux min',
                'Taux d\'absentéisme min'
            ],
            'Valeur': [
                f"{taux_global:.2f}%",
                section_max,
                f"{taux_max:.2f}%",
                section_min,
                f"{taux_min:.2f}%"
            ]
        })
        df_stats = pd.concat([df_stats, df_stats_abs], ignore_index=True)
        # === 2️⃣ Excel en mémoire ===
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Données brutes', index=False)
            ssheet_name = 'Statistiques'
            top_5.to_excel(writer, sheet_name=ssheet_name, index=False, startrow=0, startcol=0)
            startcol_prime = len(top_5.columns) + 3
            df_stats.to_excel(writer, sheet_name=ssheet_name, index=False, startrow=0, startcol=startcol_prime)

            # Feuille Graphiques
            workbook = writer.book
            worksheet_graphs = workbook.add_worksheet('Graphiques')
            writer.sheets['Graphiques'] = worksheet_graphs

            # Liste des figures
            img_bufs = []
            fig_list = [fig for fig in [fig_pr, fig_hs, fig_abs] if fig is not None]

            # Sauvegarde en mémoire
            for fig in fig_list:
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                img_bufs.append(buf)

            # Insérer en grille 2 colonnes
            row = 1
            col = 1
            max_col = 2  # Nombre de colonnes

            for i, buf in enumerate(img_bufs):
                worksheet_graphs.insert_image(row, col, f'graph_{i + 1}.png', {
                    'image_data': buf,
                    'x_scale': 0.8,
                    'y_scale': 0.8
                })

                # Passer à la colonne suivante
                col += 8  # Ajuste la largeur pour que ça ne chevauche pas

                # Si on a rempli une ligne, retour à colonne 1, nouvelle ligne
                if (i + 1) % max_col == 0:
                    col = 1
                    row += 20  # Décale vers le bas

        output.seek(0)

        # Bouton de téléchargement
        st.download_button(
            label="📥 Télécharger le rapport complet",
            data=output,
            file_name='rapport_paie.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )




    process_paie_data(df, uploaded_file)
