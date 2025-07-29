
from utils import recherche_employe

def run(df):
    import streamlit as st
    import io

    st.header("📑 Résultat Final & Paiement")
    df_filtered = recherche_employe(df)
    # Option : utiliser le DataFrame filtré ou non
    use_filtered = st.checkbox("📊 Utiliser le filtre pour les graphiques", value=False)
    if use_filtered:
        df = df_filtered  # ✅ Pas de deux-points ici !

    df = df.iloc[:-1, :]



    # CSS pour le style des cartes
    st.markdown("""
         <style>
             .salary-card {
                 background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                 padding: 20px;
                 border-radius: 15px;
                 margin: 10px 0;
                 box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                 color: white;
                 border: 1px solid rgba(255,255,255,0.2);
             }

             .salary-card-green {
                 background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
                 padding: 20px;
                 border-radius: 15px;
                 margin: 10px 0;
                 box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                 color: white;
                 border: 1px solid rgba(255,255,255,0.2);
             }

             .salary-card-orange {
                 background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                 padding: 20px;
                 border-radius: 15px;
                 margin: 10px 0;
                 box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                 color: white;
                 border: 1px solid rgba(255,255,255,0.2);
             }

             .salary-card-blue {
                 background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                 padding: 20px;
                 border-radius: 15px;
                 margin: 10px 0;
                 box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                 color: white;
                 border: 1px solid rgba(255,255,255,0.2);
             }

             .salary-card-purple {
                 background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                 padding: 20px;
                 border-radius: 15px;
                 margin: 10px 0;
                 box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                 color: #333;
                 border: 1px solid rgba(255,255,255,0.2);
             }

             .stat-title {
                 font-size: 24px;
                 font-weight: bold;
                 margin-bottom: 15px;
                 text-align: center;
                 text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
             }

             .stat-value {
                 font-size: 18px;
                 margin: 8px 0;
                 display: flex;
                 justify-content: space-between;
                 align-items: center;
             }

             .stat-label {
                 font-weight: 600;
             }

             .stat-number {
                 font-weight: bold;
                 font-size: 20px;
             }

             .summary-card {
                 background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                 padding: 25px;
                 border-radius: 20px;
                 margin: 20px 0;
                 box-shadow: 0 12px 40px rgba(0,0,0,0.15);
                 color: #333;
                 border: 2px solid rgba(255,255,255,0.3);
             }

             .metric-container {
                 display: flex;
                 justify-content: space-around;
                 flex-wrap: wrap;
                 margin: 20px 0;
             }

             .metric-box {
                 background: rgba(255,255,255,0.2);
                 padding: 15px;
                 border-radius: 10px;
                 margin: 5px;
                 min-width: 150px;
                 text-align: center;
                 backdrop-filter: blur(10px);
             }

             .alert-card {
                 background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
                 padding: 20px;
                 border-radius: 15px;
                 margin: 15px 0;
                 color: #333;
                 border: 1px solid rgba(255,255,255,0.2);
                 box-shadow: 0 8px 32px rgba(0,0,0,0.1);
             }
         </style>
         """, unsafe_allow_html=True)

    # Fonction pour créer des statistiques stylées
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import seaborn as sns
    import matplotlib.pyplot as plt

    def create_salary_statistics(df):
        # ✅ Nettoyage des noms de colonnes
        df.columns = [col.strip() for col in df.columns]

        st.markdown('<div class="summary-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-title"> STATISTIQUES DES SALAIRES</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        stats = {}

        cols_with_titles = {
            'Salaire brut': 'Salaire brut',
            'salaire net': 'salaire net',
            'Salaire net à payer': 'Salaire net à payer'
        }

        # Couleurs CSS pour chaque carte
        colors = {
            'Salaire brut': "salary-card-blue",
            'salaire net': "salary-card-green",
            'Salaire net à payer': "salary-card-orange"
        }

        # ✅ Boucle sur les colonnes de salaire
        for col, titre in cols_with_titles.items():
            if col in df.columns:
                serie_num = pd.to_numeric(df[col], errors='coerce')
                total = serie_num.sum()
                moyenne = serie_num.mean()
                maximum = serie_num.max()
                minimum = serie_num.min()
                median = serie_num.median()

                # Carte stylisée
                st.markdown(f'''
                     <div class="{colors[col]}">
                         <h3 style="margin-bottom:10px;">{titre}</h3>
                         <div class="metric-container">
                             <div class="metric-box">
                                 <div class="stat-label">Total</div>
                                 <div class="stat-number">{total:,.0f} DH</div>
                             </div>
                             <div class="metric-box">
                                 <div class="stat-label">Moyenne</div>
                                 <div class="stat-number">{moyenne:,.0f} DH</div>
                             </div>
                             <div class="metric-box">
                                 <div class="stat-label">Médiane</div>
                                 <div class="stat-number">{median:,.0f} DH</div>
                             </div>
                             <div class="metric-box">
                                 <div class="stat-label">Maximum</div>
                                 <div class="stat-number">{maximum:,.0f} DH</div>
                             </div>
                             <div class="metric-box">
                                 <div class="stat-label">Minimum</div>
                                 <div class="stat-number">{minimum:,.0f} DH</div>
                             </div>
                         </div>
                     </div>
                     ''', unsafe_allow_html=True)

                stats[f'{col}_total'] = total
                stats[f'{col}_moyenne'] = moyenne

        # ✅ Avances et prêts
        col1, col2 = st.columns(2)

        with col1:
            if 'Avance s/salaire' in df.columns:
                avance_series = pd.to_numeric(df['Avance s/salaire'], errors='coerce')
                nb_avance = (avance_series > 0).sum()
                total_avance = avance_series.sum()

                st.markdown(f'''
                     <div class="salary-card-purple">
                         <div class="stat-title"> Avances sur Salaire</div>
                         <div class="metric-container">
                             <div class="metric-box">
                                 <div class="stat-label">Employés concernés</div>
                                 <div class="stat-number">{nb_avance}</div>
                             </div>
                             <div class="metric-box">
                                 <div class="stat-label">Montant total</div>
                                 <div class="stat-number">{total_avance:,.0f} DH</div>
                             </div>
                         </div>
                     </div>
                     ''', unsafe_allow_html=True)

        with col2:
            if 'Rbst Prêt' in df.columns:
                pret_series = pd.to_numeric(df['Rbst Prêt'], errors='coerce')
                nb_pret = (pret_series > 0).sum()
                total_pret = pret_series.sum()

                st.markdown(f'''
                     <div class="salary-card">
                         <div class="stat-title"> Remboursements Prêts</div>
                         <div class="metric-container">
                             <div class="metric-box">
                                 <div class="stat-label">Employés concernés</div>
                                 <div class="stat-number">{nb_pret}</div>
                             </div>
                             <div class="metric-box">
                                 <div class="stat-label">Montant total</div>
                                 <div class="stat-number">{total_pret:,.0f} DH</div>
                             </div>
                         </div>
                     </div>
                     ''', unsafe_allow_html=True)

        return stats

    # ✅ Appel de la fonction
    stats = create_salary_statistics(df)

    # ✅ Liste des salariés concernés
    cols_affichage = ['Matricule', 'Noms & Prénoms', 'Salaire brut', 'salaire net', 'Avance s/salaire', 'Rbst Prêt']
    cols_existants = [c for c in cols_affichage if c in df.columns]

    if 'Avance s/salaire' in df.columns:
        df_avances = df[df['Avance s/salaire'] > 0]
        if not df_avances.empty:
            st.subheader("📄 Liste employés avec avances")
            st.dataframe(df_avances[cols_existants], use_container_width=True, height=400)

    if 'Rbst Prêt' in df.columns:
        df_prets = df[df['Rbst Prêt'] > 0]
        if not df_prets.empty:
            st.subheader("📄 Liste employés avec prêts")
            st.dataframe(df_prets[cols_existants], use_container_width=True, height=400)

    # ✅ Graphe 1 : Camembert Avances / Prêts
    labels = []
    values = []

    if 'Avance s/salaire' in df.columns:
        total_avance = pd.to_numeric(df['Avance s/salaire'], errors='coerce').sum()
        if total_avance > 0:
            labels.append("Avances sur salaire")
            values.append(total_avance)

    if 'Rbst Prêt' in df.columns:
        total_pret = pd.to_numeric(df['Rbst Prêt'], errors='coerce').sum()
        if total_pret > 0:
            labels.append("Remboursements Prêts")
            values.append(total_pret)

    df_stats = pd.DataFrame([stats])
    df_stats = pd.DataFrame([stats])  # ✅ Corrigé

   import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st

# 🟢 -- Graphique séparé pour les AVANCES --
img_avances = io.BytesIO()
# Supposons que vous avez une colonne 'avances' dans votre DataFrame
if 'avances' in df.columns:
    avances_data = pd.to_numeric(df['avances'], errors='coerce').dropna()
    avances_data = avances_data[avances_data > 0]  # Exclure les valeurs nulles
    
    if not avances_data.empty:
        # Calculer les statistiques
        total_avances = avances_data.sum()
        nb_beneficiaires = len(avances_data)
        moyenne_avances = avances_data.mean()
        
        # Créer des tranches logiques pour les avances
        tranches_avances = ['0-500 DH', '500-1000 DH', '1000-2000 DH', '2000+ DH']
        counts_avances = [
            len(avances_data[(avances_data > 0) & (avances_data <= 500)]),
            len(avances_data[(avances_data > 500) & (avances_data <= 1000)]),
            len(avances_data[(avances_data > 1000) & (avances_data <= 2000)]),
            len(avances_data[avances_data > 2000])
        ]
        
        fig_avances, ax_avances = plt.subplots()
        colors_avances = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        ax_avances.pie(counts_avances, labels=tranches_avances, autopct='%1.1f%%', 
                      startangle=90, colors=colors_avances)
        ax_avances.axis('equal')
        ax_avances.set_title(f"Répartition des Avances\nTotal: {total_avances:,.0f} DH | Moyenne: {moyenne_avances:,.0f} DH")
        st.pyplot(fig_avances)
        fig_avances.savefig(img_avances, format='png', dpi=300, bbox_inches='tight')
        img_avances.seek(0)

# 🟢 -- Graphique séparé pour les PRÊTS --
img_prets = io.BytesIO()
if 'prets' in df.columns:  # ou 'prêts' selon votre nomenclature
    prets_data = pd.to_numeric(df['prets'], errors='coerce').dropna()
    prets_data = prets_data[prets_data > 0]
    
    if not prets_data.empty:
        # Calculer les statistiques
        total_prets = prets_data.sum()
        nb_beneficiaires_prets = len(prets_data)
        moyenne_prets = prets_data.mean()
        
        # Créer des tranches logiques pour les prêts (généralement plus élevés)
        tranches_prets = ['0-2000 DH', '2000-5000 DH', '5000-10000 DH', '10000+ DH']
        counts_prets = [
            len(prets_data[(prets_data > 0) & (prets_data <= 2000)]),
            len(prets_data[(prets_data > 2000) & (prets_data <= 5000)]),
            len(prets_data[(prets_data > 5000) & (prets_data <= 10000)]),
            len(prets_data[prets_data > 10000])
        ]
        
        fig_prets, ax_prets = plt.subplots()
        colors_prets = ['#ffb3ba', '#bae1ff', '#baffc9', '#ffffba']
        ax_prets.pie(counts_prets, labels=tranches_prets, autopct='%1.1f%%', 
                    startangle=90, colors=colors_prets)
        ax_prets.axis('equal')
        ax_prets.set_title(f"Répartition des Prêts\nTotal: {total_prets:,.0f} DH | Moyenne: {moyenne_prets:,.0f} DH")
        st.pyplot(fig_prets)
        fig_prets.savefig(img_prets, format='png', dpi=300, bbox_inches='tight')
        img_prets.seek(0)

# 🟢 -- Histogramme OUVRIERS (Salaire équivalent basé sur jours travaillés) --
img_hist_ouvriers = io.BytesIO()
if 'Salaire net a paye' in df.columns and 'categorie' in df.columns and 'jours_travailles' in df.columns:
    # Filtrer les ouvriers
    ouvriers_df = df[df['categorie'].str.lower().str.contains('ouvrier', na=False)]
    
    if not ouvriers_df.empty:
        # Utiliser le "Salaire net a paye" qui est le montant réellement perçu
        ouvriers_df = ouvriers_df.copy()
        ouvriers_df['salaire_paye_num'] = pd.to_numeric(ouvriers_df['Salaire net a paye'], errors='coerce')
        ouvriers_df['jours_num'] = pd.to_numeric(ouvriers_df['jours_travailles'], errors='coerce')
        
        # Calculer le salaire journalier réel
        ouvriers_df['salaire_journalier'] = ouvriers_df['salaire_paye_num'] / ouvriers_df['jours_num']
        
        # Calculer le salaire équivalent 30 jours (capacité de gain mensuel)
        ouvriers_df['salaire_equivalent_30j'] = ouvriers_df['salaire_journalier'] * 30
        
        # Filtrer les données valides et >= SMIC
        sal_ouvriers = ouvriers_df['salaire_equivalent_30j'].dropna()
        sal_ouvriers = sal_ouvriers[sal_ouvriers >= 3000]
        
        if not sal_ouvriers.empty:
            fig_hist_ouv, ax_hist_ouv = plt.subplots(figsize=(10, 6))
            
            # Créer des bins logiques à partir du SMIC
            bins = np.arange(3000, sal_ouvriers.max() + 500, 500)
            
            ax_hist_ouv.hist(sal_ouvriers, bins=bins, color='#4facfe', edgecolor='black', alpha=0.7)
            ax_hist_ouv.axvline(x=3000, color='red', linestyle='--', linewidth=2, label='SMIC (3000 DH)')
            ax_hist_ouv.axvline(x=sal_ouvriers.mean(), color='orange', linestyle='-', linewidth=2, 
                               label=f'Moyenne: {sal_ouvriers.mean():.0f} DH')
            
            ax_hist_ouv.set_title('Distribution des Salaires Ouvriers\n(Capacité de gain équivalent 30 jours)')
            ax_hist_ouv.set_xlabel('Salaire Net Équivalent 30j (DH)')
            ax_hist_ouv.set_ylabel('Nombre d\'Ouvriers')
            ax_hist_ouv.legend()
            ax_hist_ouv.grid(True, alpha=0.3)
            
            # Ajouter des infos contextuelles
            textstr = f'Basé sur: Salaire net à payer / Jours travaillés × 30\nÉchantillon: {len(sal_ouvriers)} ouvriers'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax_hist_ouv.text(0.02, 0.98, textstr, transform=ax_hist_ouv.transAxes, fontsize=9,
                            verticalalignment='top', bbox=props)
            
            st.pyplot(fig_hist_ouv)
            fig_hist_ouv.savefig(img_hist_ouvriers, format='png', dpi=300, bbox_inches='tight')
            img_hist_ouvriers.seek(0)

# 🟢 -- Histogramme EMPLOYÉS --
img_hist_employes = io.BytesIO()
if 'salaire net' in df.columns and 'categorie' in df.columns:
    # Filtrer les employés
    employes_df = df[df['categorie'].str.lower().str.contains('employe|employé', na=False)]
    
    if not employes_df.empty:
        employes_df = employes_df.copy()
        # Utiliser le "salaire net" pour les employés (salaire contractuel mensuel)
        employes_df['salaire_net_num'] = pd.to_numeric(employes_df['salaire net'], errors='coerce')
        
        # Pour les employés, prendre le salaire tel quel (généralement mensuel)
        sal_employes = employes_df['salaire_net_num'].dropna()
        sal_employes = sal_employes[sal_employes >= 3000]
        
        if not sal_employes.empty:
            fig_hist_emp, ax_hist_emp = plt.subplots(figsize=(10, 6))
            
            # Bins adaptés aux salaires d'employés (généralement plus élevés)
            bins = np.arange(3000, sal_employes.max() + 1000, 1000)
            
            ax_hist_emp.hist(sal_employes, bins=bins, color='#2ecc71', edgecolor='black', alpha=0.7)
            ax_hist_emp.axvline(x=3000, color='red', linestyle='--', linewidth=2, label='SMIC (3000 DH)')
            ax_hist_emp.axvline(x=sal_employes.mean(), color='orange', linestyle='-', linewidth=2, 
                               label=f'Moyenne: {sal_employes.mean():.0f} DH')
            
            ax_hist_emp.set_title('Distribution des Salaires Employés\n(Salaire contractuel mensuel)')
            ax_hist_emp.set_xlabel('Salaire Net Contractuel (DH)')
            ax_hist_emp.set_ylabel('Nombre d\'Employés')
            ax_hist_emp.legend()
            ax_hist_emp.grid(True, alpha=0.3)
            
            # Ajouter des infos contextuelles
            textstr = f'Basé sur: Salaire net contractuel\nÉchantillon: {len(sal_employes)} employés'
            props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
            ax_hist_emp.text(0.02, 0.98, textstr, transform=ax_hist_emp.transAxes, fontsize=9,
                            verticalalignment='top', bbox=props)
            
            st.pyplot(fig_hist_emp)
            fig_hist_emp.savefig(img_hist_employes, format='png', dpi=300, bbox_inches='tight')
            img_hist_employes.seek(0)

# 🟢 -- Statistiques récapitulatives --
st.subheader("📊 Résumé des Statistiques")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if 'avances_data' in locals() and not avances_data.empty:
        st.metric("Total Avances", f"{total_avances:,.0f} DH", f"{nb_beneficiaires} bénéficiaires")

with col2:
    if 'prets_data' in locals() and not prets_data.empty:
        st.metric("Total Prêts", f"{total_prets:,.0f} DH", f"{nb_beneficiaires_prets} bénéficiaires")

with col3:
    if 'sal_ouvriers' in locals() and not sal_ouvriers.empty:
        st.metric("Salaire Moyen Ouvriers", f"{sal_ouvriers.mean():.0f} DH", f"{len(sal_ouvriers)} personnes")

with col4:
    if 'sal_employes' in locals() and not sal_employes.empty:
        st.metric("Salaire Moyen Employés", f"{sal_employes.mean():.0f} DH", f"{len(sal_employes)} personnes")
    # ✅ Générer le Excel avec plusieurs feuilles
    # -----------------------------
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:

        df_stats.to_excel(writer, sheet_name='Statistiques', index=False)


        if not df_avances.empty:
            df_avances.to_excel(writer, sheet_name='Avances', index=False)


        if not df_prets.empty:
            df_prets.to_excel(writer, sheet_name='Prets', index=False)


        workbook = writer.book
        worksheet = workbook.add_worksheet('Graphiques')
        writer.sheets['Graphiques'] = worksheet

        # Insérer camembert SI NON VIDE
        if values:
            worksheet.insert_image('B2', 'camembert.png', {'image_data': img_camembert})

        # Insérer histogramme SI NON VIDE
        if 'salaire net' in df.columns and not sal_net_series.empty:
            worksheet.insert_image('B25', 'histogramme.png', {'image_data': img_hist})

    output.seek(0)

    # === ✅ BOUTON DE TÉLÉCHARGEMENT ===
    st.download_button(
        "📥 Télécharger le fichier Excel complet",
        data=output,
        file_name="rapport_salaire_complet.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )




