import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings

# --- Dicionário de Tradução (i18n) ---
# (ATUALIZADO V40)
TEXT_DICT = {
    'pt': {
        'app_title': "🔬 Analisador de Liberação de Fármacos (Células de Franz) v40",
        'sidebar_nav': "Navegação",
        'sidebar_info': "App V40: Correção de compatibilidade (TypeError: include_groups).",
        # ... (Restante dos textos idêntico ao V26) ...
        'nav_step1': "Etapa 1: Configuração e Upload",
        'nav_step2': "Etapa 2: Dados Processados e Agregados",
        'nav_step3': "Etapa 3: Gráficos de Liberação",
        'nav_step4': "Etapa 4: Modelagem Cinética",
        'nav_step5': "Etapa 5: Resumo Comparativo",
        'nav_step6': "Etapa 6: Análise por IA (Gerar Prompt)", 
        
        'step1_header': "Etapa 1: Configuração e Upload 📈",
        'step1_subheader_units': "Unidades Globais",
        'step1_label_vol_unit': "Unidade de Volume",
        'step1_label_conc_unit': "Unidade de Concentração",
        'step1_subheader_cell': "Parâmetros da Célula (Global)",
        'step1_label_vol_receptor': "Volume Receptor",
        'step1_label_vol_sample': "Volume da Amostra Coletada",
        'step1_subheader_calib': "Curva de Calibração (Global)",
        'step1_label_calib_a': "Coeficiente Angular (a)",
        'step1_label_calib_b': "Intercepto (b)",
        'step1_help_calib_b': "Valores negativos são permitidos.",
        'step1_subheader_time': "Configuração do Tempo (Planilha)",
        'step1_label_time_unit': "Unidade de Tempo no Cabeçalho",
        'step1_time_hours': "Horas",
        'step1_time_minutes': "Minutos",
        'step1_subheader_upload': "Carregar Dados do Experimento (Formato Largo)",
        'step1_expander_model': "Ver Modelo de Referência da Planilha",
        'step1_uploader_label': "Carregue seu arquivo",
        'step1_csv_options': "Opções de leitura de CSV/TXT detectadas:",
        'step1_csv_sep': "Separador de Coluna",
        'step1_csv_sep_semi': "; (Ponto e Vírgula)",
        'step1_csv_sep_comma': ", (Vírgula)",
        'step1_csv_sep_tab': "Tab",
        'step1_csv_dec': "Separador Decimal",
        'step1_csv_dec_comma': ", (Vírgula)",
        'step1_csv_dec_point': ". (Ponto)",
        'step1_error_calib_a': "O Coeficiente Angular (a) da calibração não pode ser zero.",
        'step1_upload_preview': "Dados Carregados (Pré-visualização):",
        'step1_subheader_dose': "Configurações da Formulação (Dose por Grupo)",
        'step1_dose_group': "Grupo:",
        'step1_dose_conc': "Concentração",
        'step1_dose_vol': "Volume Adicionado",
        'step1_button_process': "Processar Dados",
        'step1_spinner_process': "Processando...",
        'step1_success_process': "Dados processados! Navegue para a 'Etapa 2'.",
        'step1_error_process': "Erro ao ler ou processar o arquivo:",
        'step1_col_conc_name': "Concentracao",
        'step1_col_q_name': "Q_Acumulada_Corrigida",
        'step1_col_pct_name': "Percent_Liberado (%)",
        'step1_y_label_pct': "Fármaco Liberado (%)",
        'step1_y_label_q': "Quantidade Acumulada",

        'step2_header': "Etapa 2: Análise de Dados Processados 📊",
        'step2_warning_process': "Por favor, carregue e processe os dados na 'Etapa 1' primeiro.",
        'step2_info': "Resultados dos cálculos. A 'Tabela Agregada' é usada para gráficos e modelagem.",
        'step2_subheader_raw': "1. Tabela de Dados Processados (Por Réplica)",
        'step2_subheader_agg': "2. Tabela de Dados Agregados (Média e Desvio Padrão)",

        'step3_header': "Etapa 3: Gráficos de Liberação 📈",
        'step3_info_export': "Use o ícone de câmera 📷 no canto superior direito dos gráficos (ao passar o mouse) para exportá-los como PNG.",
        'step3_subheader_individual': "1. Análise de Grupo Individual (com Desvio Padrão)",
        'step3_selectbox_group': "Selecione o Grupo para Análise:",
        'step3_legend_individual': "Média (± SD)",
        'step3_title_individual': "Perfil de Liberação (Média ± SD) para:",
        'step3_xaxis_label': "Tempo (horas)",
        'step3_subheader_compare': "2. Gráfico Comparativo (Médias de Todos os Grupos com Desvio Padrão)",
        'step3_title_compare': "Comparação dos Perfis de Liberação (Média ± SD)",
        'step3_color_picker_label': "Personalizar Cores (Gráfico Comparativo)",
        'step3_color_picker_group': "Cor para",

        'step4_header': "Etapa 4: Modelagem Cinética 🔬",
        'step4_info_model': "A modelagem é realizada sobre os dados de **Média** do grupo selecionado.",
        'step4_selectbox_group': "Selecione o Grupo para Modelagem:",
        'step4_error_points': "Pontos de dados insuficientes (mínimo 3, excluindo t=0) para modelar.",
        'step4_error_fit': "Nenhum modelo pôde ser ajustado.",
        'step4_subheader_results': "Resultados da Modelagem para:",
        'step4_subheader_interp': "Interpretação Automática (Detalhada)",
        'step4_success_model': "O modelo com melhor ajuste (maior R²) é:",
        'step4_interp_params': "Parâmetros:",
        'step4_interp_const': "Constante (k ou a):",
        'step4_interp_exp': "Expoente (n ou b):",
        'step4_interp_interp': "Interpretação:",
        'step4_glossary_header': "Glossário do Mecanismo",
        'step4_glossary_fickian': "Significado: A taxa de liberação é controlada pela difusão do fármaco através da matriz (ex: gel, polímero). A velocidade de liberação diminui com o tempo à medida que a concentração de fármaco diminui. É proporcional à raiz quadrada do tempo (Modelo de Higuchi).",
        'step4_glossary_anomalous': "Significado: A liberação é controlada por uma *combinação* de dois mecanismos: (1) a difusão do fármaco (Fickiana) e (2) o inchaço (swelling) ou relaxamento das cadeias do polímero. É comum em hidrogéis e sistemas poliméricos.",
        'step4_glossary_case_ii': "Significado: A liberação é dominada *inteiramente* pelo relaxamento ou erosão da matriz polimérica. A difusão é muito rápida em comparação. Isso geralmente resulta em uma liberação de Ordem Zero (taxa constante).",
        'step4_glossary_zero_order': "Significado: A liberação ocorre a uma taxa constante, liberando a *mesma quantidade de fármaco por unidade de tempo*, independente da concentração restante. É o ideal para liberação controlada.",
        'step4_glossary_first_order': "Significado: A taxa de liberação é *dependente da concentração*. Ela é rápida no início (quando a concentração é alta) e diminui exponencialmente à medida que o fármaco é liberado. Típico de sistemas onde o fármaco está dissolvido na matriz.",
        'step4_glossary_hixson': "Significado: Este modelo descreve a liberação de fármacos onde a *área de superfície* do sistema diminui com o tempo. É típico de sistemas que se dissolvem ou erodem uniformemente (ex: comprimidos que se dissolvem).",
        'step4_glossary_complex': "Significado: A liberação não segue um único mecanismo claro, but sim uma combinação de vários processos (ex: difusão, erosão, inchaço) de forma simultânea.",
        'step4_glossary_select': "Selecione um grupo para ver a interpretação do mecanismo.",

        'step5_header': "Etapa 5: Resumo Comparativo 🏆",
        'step5_info': "Abaixo está um resumo do melhor modelo cinético para cada grupo, baseado no maior R².",
        'step5_spinner': "Analisando todos os grupos...",
        'step5_col_group': "Grupo",
        'step5_col_model': "Melhor_Modelo",
        'step5_col_r2': "R2",
        'step5_col_k': "Constante (k ou a)",
        'step5_col_k_unit': "Unidade_Constante",
        'step5_col_n': "Expoente (n ou b)",
        'step5_col_interp': "Interpretação",
        'step5_error_data': "Dados insuficientes.",
        
        'model_zero_order': "Ordem Zero",
        'model_first_order': "Primeira Ordem",
        'model_higuchi': "Higuchi",
        'model_korsmeyer': "Korsmeyer-Peppas",
        'model_hixson': "Hixson-Crowell",
        'model_weibull': "Weibull",
        'interp_fail': "Falha na modelagem.",
        'interp_r2_fail': "Falha ao determinar R².",
        'interp_fickian': "Difusão Fickiana",
        'interp_anomalous': "Transporte Anômalo (Não-Fickiano)",
        'interp_case_ii': "Transporte Caso-II (Zero Ordem)",
        'interp_zero_order_mech': "Liberação a taxa constante",
        'interp_first_order_mech': "Liberação dependente da concentração",
        'interp_hixson_mech': "Liberação por dissolução/erosão da matriz",
        'interp_complex': "Mecanismo complexo/combinado",
        
        'step6_header': "Etapa 6: Análise por IA (Gerador de Prompt) 🤖",
        'step6_info': "Esta etapa prepara um 'briefing' completo com seus dados. Copie o texto gerado e cole-o em um chat com uma IA (como o Gemini) para obter uma análise detalhada, pesquisa de literatura e redação de resultados.",
        'step6_subheader_context': "1. Forneça o Contexto do Estudo",
        'step6_label_drug': "Nome do Fármaco",
        'step6_label_system': "Veículo ou Sistema (ex: 'nanopartículas de PLGA', 'gel de quitosana')",
        'step6_label_objective': "Principal Objetivo ou Comparação (ex: 'Comparar F1 vs F2', 'Avaliar efeito do polímero')",
        'step6_label_medium': "Solução de Liberação (Meio)", 
        'step6_label_membrane': "Membrana Utilizada (ex: acetato de celulose)", 
        'step6_button_generate': "Gerar Prompt de Análise para IA",
        'step6_prompt_header': "Seu Prompt de IA (Pronto para Copiar):",
        'step6_copy_success': "Prompt gerado com sucesso! Copie o texto abaixo e cole no seu chat de IA.",
        
        'prompt_title': "### Análise de Estudo de Liberação de Fármaco (Células de Franz)",
        'prompt_context_header': "### 1. Contexto do Estudo",
        'prompt_context_drug': "Fármaco",
        'prompt_context_system': "Sistema/Veículo",
        'prompt_context_objective': "Objetivo Principal",
        'prompt_methods_header': "### 2. Parâmetros Metodológicos (Resumido)",
        'prompt_methods_vol_receptor': "Volume da Célula Receptora",
        'prompt_methods_vol_sample': "Volume da Amostra Coletada",
        'prompt_methods_medium': "Meio de Liberação", 
        'prompt_methods_membrane': "Membrana", 
        'prompt_results_header': "### 3. Resultados de Liberação (Ponto Final)",
        'prompt_kinetics_header': "### 4. Resultados da Modelagem Cinética",
        'prompt_tasks_header': "### 5. Tarefas Solicitadas",
        'prompt_task_1': "1. **Redação de Metodologia:** Com base nos parâmetros, escreva uma seção de 'Metodologia' em formato de artigo científico para o ensaio de liberação *in vitro*.",
        'prompt_task_2': "2. **Pesquisa e Tabela de Literatura:** Com base no Fármaco e Sistema, pesquise na literatura por estudos semelhantes. Crie uma tabela comparando os resultados da literatura (especialmente o mecanismo e a % de liberação) com os meus 'Resultados da Modelagem Cinética' (Tabela 4). **Importante: Inclua o DOI ou Link para cada artigo na tabela.**", 
        'prompt_task_3': "3. **Discussão dos Resultados:** Escreva uma 'Discussão' em formato de artigo. Analise os dados das Tabelas 3 e 4, explique o que o 'Melhor Modelo' (ex: Higuchi) significa para cada formulação e compare os grupos entre si (e com a literatura da Tarefa 2), focando no 'Objetivo Principal'.",
    },
    'en': {
        # ... (Dicionário 'en' completo) ...
        'app_title': "🔬 Drug Release Analyzer (Franz Cells) v40",
        'sidebar_nav': "Navigation",
        'sidebar_info': "App V40: Compatibility fix (TypeError: include_groups).",
        # ... (Restante idêntico ao V26) ...
    }
}

# --- Função de Carregamento de Dados (V27) ---
def load_data(uploaded_file, sep, decimal):
    """Tenta ler um CSV/TXT com codificações comuns."""
    try:
        # 1. Tentar com UTF-8 (padrão)
        return pd.read_csv(uploaded_file, sep=sep, decimal=decimal, encoding='utf-8')
    except UnicodeDecodeError:
        # 2. Se falhar, rebobinar o arquivo e tentar com Latin-1
        st.warning("Falha ao ler com UTF-8. Tentando com codificação 'latin-1'...")
        uploaded_file.seek(0) # Crucial: rebobina o arquivo para o início
        return pd.read_csv(uploaded_file, sep=sep, decimal=decimal, encoding='latin-1')
    except Exception as e:
        # Outros erros de leitura
        st.error(f"Erro inesperado ao ler o arquivo: {e}")
        return None

# --- Funções dos Modelos Cinéticos (Idênticas V31) ---
def model_zero_order(t, k0): return k0 * t
def model_first_order(t, Q_inf, k1): return Q_inf * (1 - np.exp(-k1 * t))
def model_higuchi(t, kH): return kH * np.sqrt(t)
def model_korsmeyer_peppas(t, kKP, n): return kKP * (t**n)
def model_hixson_crowell(t, Q_inf, kHC):
    termo = 1 - kHC * t
    termo = np.where(termo < 0, 0, termo)
    return Q_inf * (1 - (termo)**3)
def model_weibull(t, Q_inf, a, b):
    t_a = t / a
    t_a = np.where(t_a <= 0, 1e-9, t_a)
    return Q_inf * (1 - np.exp(-(t_a**b)))

# --- Função de Cálculo V9 (Com Saturação 100%) (Idêntica V31) ---
def calcular_liberacao_replica_v9(df_group, col_grupo, vol_celula, vol_amostra, cal_a, cal_b, doses_dict, col_conc, col_q_acumulada, col_percent):
    df_group = df_group.sort_values(by='Tempo')
    grupo_da_replica = df_group[col_grupo].iloc[0]
    dose_total = doses_dict.get(grupo_da_replica, {}).get('dose_total', 0.0)
    df_group[col_conc] = (df_group['Area'] - cal_b) / cal_a
    
    df_group[col_conc] = df_group[col_conc].fillna(0)

    correcao_acumulada = 0
    q_acumulada_corrigida = []
    
    for i in range(len(df_group)):
        conc_atual = df_group.iloc[i][col_conc]
        
        if pd.isna(conc_atual):
            conc_atual = 0.0

        q_no_recipiente = conc_atual * vol_celula
        q_corrigida = q_no_recipiente + correcao_acumulada
        q_acumulada_corrigida.append(q_corrigida)
        correcao_acumulada += (conc_atual * vol_amostra)
        
    df_group[col_q_acumulada] = q_acumulada_corrigida
    
    if dose_total > 0:
        df_group[col_percent] = (df_group[col_q_acumulada] / dose_total) * 100
        df_group[col_percent] = df_group[col_percent].clip(lower=0)
        
        idx_100_series = (df_group[col_percent] > 99.9999)
        if idx_100_series.any():
            first_idx_100 = idx_100_series.idxmax()
            df_group.loc[first_idx_100:, col_percent] = 100.0
            
    else:
        df_group[col_percent] = 0.0
        
    return df_group

# --- NOVO V38: Função de Modelagem V11 (Linha Completa Limpa) ---
def rodar_modelagem_v11(t_data, q_data, df_model, y_axis_mean, has_dose_info):
    resultados_df_list = []
    modeling_messages = []
    
    # Limite estatístico de significância: R² < 0.05 é considerado insignificante.
    R2_THRESHOLD = 0.05
    
    q_max = q_data.max()
    if q_max == 0: q_max = 1.0
    t_data = np.array(t_data)
    q_data = np.array(q_data)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # --- Lógica KP V32/V35 ---
        limite_kp = 60.0 if has_dose_info else q_max * 0.6
        
        df_kp = df_model[df_model[y_axis_mean] <= limite_kp]
        t_data_kp = df_kp["Tempo"]
        q_data_kp = df_kp[y_axis_mean]

        # Modelo 4: Korsmeyer-Peppas (kKP > 0, n > 0)
        if len(t_data_kp) >= 3:
            try:
                popt_kp, _ = curve_fit(model_korsmeyer_peppas, t_data_kp, q_data_kp, 
                                       p0=[q_max*0.1, 0.5], 
                                       maxfev=5000,
                                       bounds=([0, 0], [np.inf, 2.0]))
                r2_kp = r2_score(q_data_kp, model_korsmeyer_peppas(t_data_kp, *popt_kp))
                
                if r2_kp >= R2_THRESHOLD:
                    resultados_df_list.append({"Modelo": "Korsmeyer-Peppas", "R2": r2_kp, "kKP": popt_kp[0], "n": popt_kp[1]})
                else:
                    # R2 insignificante: Limpa a linha inteira (V38)
                    resultados_df_list.append({"Modelo": "Korsmeyer-Peppas", "R2": np.nan, "kKP": np.nan, "n": np.nan})
                    modeling_messages.append(f"**Korsmeyer-Peppas:** Ajuste realizado, mas $R^2={r2_kp:.4f}$ é insignificante ($<0.05$).")
            except (RuntimeError, ValueError) as e:
                # Falha total: Limpa a linha inteira (V38)
                resultados_df_list.append({"Modelo": "Korsmeyer-Peppas", "R2": np.nan, "kKP": np.nan, "n": np.nan})
                modeling_messages.append(f"**Korsmeyer-Peppas:** O ajuste falhou. Causa: Não convergiu para uma solução.")
        else:
            # Caso "Pulado" - Apenas adiciona a mensagem, não polui o DF de resultados
            modeling_messages.append(f"**Korsmeyer-Peppas:** Modelo pulado. Pontos insuficientes (N={len(t_data_kp)}) abaixo do limite de 60% da liberação total, que é o requisito para este modelo.")
        
        # --- Outros Modelos (rodam nos dados completos) ---
        
        # Modelo 1: Ordem Zero (k0 > 0)
        try:
            popt_zo, _ = curve_fit(model_zero_order, t_data, q_data, bounds=([0], [np.inf]), maxfev=10000)
            r2_zo = r2_score(q_data, model_zero_order(t_data, *popt_zo))
            if r2_zo >= R2_THRESHOLD:
                resultados_df_list.append({"Modelo": "Zero-Order", "R2": r2_zo, "k0": popt_zo[0]})
            else:
                # R2 insignificante: Limpa a linha inteira (V38)
                resultados_df_list.append({"Modelo": "Zero-Order", "R2": np.nan, "k0": np.nan})
                modeling_messages.append(f"**Ordem Zero:** Ajuste realizado, mas $R^2={r2_zo:.4f}$ é insignificante ($<0.05$).")
        except (RuntimeError, ValueError) as e:
            # Falha total: Limpa a linha inteira (V38)
            resultados_df_list.append({"Modelo": "Zero-Order", "R2": np.nan, "k0": np.nan})
            modeling_messages.append(f"**Ordem Zero:** O ajuste falhou. Causa: Não convergiu para uma solução.")

        # Modelo 2: Primeira Ordem (Q_inf > 0, k1 > 0)
        try:
            popt_fo, _ = curve_fit(model_first_order, t_data, q_data, 
                                   p0=[q_max, 0.1], 
                                   bounds=([0, 0], [np.inf, np.inf]), maxfev=10000)
            r2_fo = r2_score(q_data, model_first_order(t_data, *popt_fo))
            if r2_fo >= R2_THRESHOLD:
                resultados_df_list.append({"Modelo": "First-Order", "R2": r2_fo, "k1": popt_fo[1]})
            else:
                # R2 insignificante: Limpa a linha inteira (V38)
                resultados_df_list.append({"Modelo": "First-Order", "R2": np.nan, "k1": np.nan})
                modeling_messages.append(f"**Primeira Ordem:** Ajuste realizado, mas $R^2={r2_fo:.4f}$ é insignificante ($<0.05$).")
        except (RuntimeError, ValueError) as e:
            # Falha total: Limpa a linha inteira (V38)
            resultados_df_list.append({"Modelo": "First-Order", "R2": np.nan, "k1": np.nan})
            modeling_messages.append(f"**Primeira Ordem:** O ajuste falhou. Causa: Não convergiu para uma solução.")

        # Modelo 3: Higuchi (kH > 0)
        try:
            popt_h, _ = curve_fit(model_higuchi, t_data, q_data, bounds=([0], [np.inf]), maxfev=10000)
            r2_h = r2_score(q_data, model_higuchi(t_data, *popt_h))
            if r2_h >= R2_THRESHOLD:
                resultados_df_list.append({"Modelo": "Higuchi", "R2": r2_h, "kH": popt_h[0]})
            else:
                # R2 insignificante: Limpa a linha inteira (V38)
                resultados_df_list.append({"Modelo": "Higuchi", "R2": np.nan, "kH": np.nan})
                modeling_messages.append(f"**Higuchi:** Ajuste realizado, mas $R^2={r2_h:.4f}$ é insignificante ($<0.05$).")
        except (RuntimeError, ValueError) as e:
            # Falha total: Limpa a linha inteira (V38)
            resultados_df_list.append({"Modelo": "Higuchi", "R2": np.nan, "kH": np.nan})
            modeling_messages.append(f"**Higuchi:** O ajuste falhou. Causa: Não convergiu para uma solução.")

        # Modelo 5: Hixson-Crowell (Q_inf > 0, kHC > 0)
        try:
            popt_hc, _ = curve_fit(model_hixson_crowell, t_data, q_data, 
                                   p0=[q_max, 0.01], maxfev=10000,
                                   bounds=([0, 0], [np.inf, np.inf])) 
            r2_hc = r2_score(q_data, model_hixson_crowell(t_data, *popt_hc))
            if r2_hc >= R2_THRESHOLD:
                resultados_df_list.append({"Modelo": "Hixson-Crowell", "R2": r2_hc, "kHC": popt_hc[1]})
            else:
                # R2 insignificante: Limpa a linha inteira (V38)
                resultados_df_list.append({"Modelo": "Hixson-Crowell", "R2": np.nan, "kHC": np.nan})
                modeling_messages.append(f"**Hixson-Crowell:** Ajuste realizado, mas $R^2={r2_hc:.4f}$ é insignificante ($<0.05$).")
        except (RuntimeError, ValueError) as e:
            # Falha total: Limpa a linha inteira (V38)
            resultados_df_list.append({"Modelo": "Hixson-Crowell", "R2": np.nan, "kHC": np.nan})
            modeling_messages.append(f"**Hixson-Crowell:** O ajuste falhou. Causa: Não convergiu para uma solução.")

        # Modelo 6: Weibull (Q_inf > 0, a > 0, b > 0)
        try:
            popt_w, _ = curve_fit(model_weibull, t_data, q_data, 
                                  p0=[q_max, 2, 1], maxfev=10000,
                                  bounds=([0, 1e-9, 1e-9], [np.inf, np.inf, np.inf])) 
            r2_w = r2_score(q_data, model_weibull(t_data, *popt_w))
            if r2_w >= R2_THRESHOLD:
                resultados_df_list.append({"Modelo": "Weibull", "R2": r2_w, "a": popt_w[1], "b": popt_w[2]})
            else:
                # R2 insignificante: Limpa a linha inteira (V38)
                resultados_df_list.append({"Modelo": "Weibull", "R2": np.nan, "a": np.nan, "b": np.nan})
                modeling_messages.append(f"**Weibull:** Ajuste realizado, mas $R^2={r2_w:.4f}$ é insignificante ($<0.05$).")
        except (RuntimeError, ValueError) as e:
            # Falha total: Limpa a linha inteira (V38)
            resultados_df_list.append({"Modelo": "Weibull", "R2": np.nan, "a": np.nan, "b": np.nan})
            modeling_messages.append(f"**Weibull:** O ajuste falhou. Causa: Não convergiu para uma solução.")


    if not resultados_df_list:
        modeling_messages.append("Nenhum modelo cinético pôde ser ajustado com sucesso. Verifique seus dados (pontos insuficientes ou liberação muito baixa).")
        return pd.DataFrame(), modeling_messages
        
    df_resultados = pd.DataFrame(resultados_df_list).set_index("Modelo").fillna(np.nan)
    
    return df_resultados, modeling_messages

# --- Função de Interpretação (Corrigida V39: Tratamento de KeyError) ---
def interpretar_resultados(df_resultados, y_label, lang_key):
    T = TEXT_DICT[lang_key]
    base_unit = "%" if "%" in y_label else y_label.split('(')[-1].replace(')', '')
    time_unit = "h"
    
    # 1. Checagem inicial de falha
    if df_resultados.empty or 'R2' not in df_resultados.columns or df_resultados['R2'].isnull().all():
        return "N/A", 0.0, np.nan, "-", np.nan, T['interp_fail'], "N/A"
    
    try:
        # 2. Seleciona o R2 máximo ignorando os NaNs (falhas/insignificantes)
        r2_series = df_resultados["R2"].astype(float).dropna()
        if r2_series.empty:
            return "N/A", 0.0, np.nan, "-", np.nan, T['interp_fail'], "N/A"
        
        melhor_modelo_key = r2_series.idxmax()
        melhor_modelo_str = T.get(f'model_{melhor_modelo_key.lower().replace("-","_")}', melhor_modelo_key)
        r2_melhor = df_resultados.loc[melhor_modelo_key, 'R2']
    
    except Exception as e:
        # Se ocorrer qualquer erro na seleção do melhor modelo (incluindo KeyError ou erro de índice)
        st.error(f"Erro interno ao determinar o melhor modelo: {e}")
        return "N/A", 0.0, np.nan, "-", np.nan, T['interp_r2_fail'], "N/A"
    
    interpretacao_mecanismo = ""
    glossario_key = "N/A" 
    k_val = np.nan
    k_unit = "-"
    n_val = np.nan
    
    # 3. Extração dos Parâmetros e Interpretação
    try:
        if melhor_modelo_key == "Korsmeyer-Peppas":
            n_val = df_resultados.loc[melhor_modelo_key, "n"]
            k_val = df_resultados.loc[melhor_modelo_key, "kKP"]
            k_unit = f"{base_unit}/{time_unit}^n"
            # Adiciona uma checagem se o valor é válido antes de tentar formatá-lo/usá-lo
            if pd.notna(n_val):
                interpretacao_mecanismo = f"**{T['step4_interp_exp']}** {n_val:.3f}. "
                if n_val < 0.45: glossario_key = "fickian"
                elif 0.45 <= n_val < 0.89: glossario_key = "anomalous"
                elif n_val >= 0.89: glossario_key = "case_ii"
            interpretacao_mecanismo += f"Mecanismo: {T[f'interp_{glossario_key}']}."
        
        elif melhor_modelo_key == "Higuchi":
            k_val = df_resultados.loc[melhor_modelo_key, "kH"]
            k_unit = f"{base_unit}/{time_unit}^0.5"
            glossario_key = "fickian"
            interpretacao_mecanismo = T['interp_fickian']

        elif melhor_modelo_key == "Zero-Order":
            k_val = df_resultados.loc[melhor_modelo_key, "k0"]
            k_unit = f"{base_unit}/{time_unit}"
            glossario_key = "zero_order"
            interpretacao_mecanismo = f"{T['interp_zero_order_mech']} (k0 = {k_val:.4f} {k_unit})."

        elif melhor_modelo_key == "First-Order":
            k_val = df_resultados.loc[melhor_modelo_key, "k1"]
            k_unit = f"1/{time_unit}"
            glossario_key = "first_order"
            interpretacao_mecanismo = T['interp_first_order_mech']

        elif melhor_modelo_key == "Hixson-Crowell":
            k_val = df_resultados.loc[melhor_modelo_key, "kHC"]
            k_unit = f"1/{time_unit}"
            glossario_key = "hixson"
            interpretacao_mecanismo = T['interp_hixson_mech']

        elif melhor_modelo_key == "Weibull":
            n_val = df_resultados.loc[melhor_modelo_key, "b"]
            k_val = df_resultados.loc[melhor_modelo_key, "a"]
            k_unit = time_unit
            if pd.notna(n_val):
                interpretacao_mecanismo = f"Expoente de Forma (b) = {n_val:.3f}. "
                if n_val < 0.75: glossario_key = "fickian"
                elif 0.75 <= n_val <= 1.0: glossario_key = "complex"
                elif n_val > 1.0: glossario_key = "case_ii"
            interpretacao_mecanismo += f"Mecanismo: {T[f'interp_{glossario_key}']}."
    
    except KeyError:
        # Pega a exceção se a coluna específica (kKP, n, kH) não existir no resultado final
        st.warning(f"Aviso: Parâmetros específicos (k, n) para o modelo '{melhor_modelo_key}' não foram encontrados. Isso é esperado se o modelo tiver falhado no cálculo dos parâmetros.")
        return melhor_modelo_str, r2_melhor, np.nan, "-", np.nan, interpretacao_mecanismo, glossario_key
        
    return melhor_modelo_str, r2_melhor, k_val, k_unit, n_val, interpretacao_mecanismo, glossario_key

# --- Função Glossário (Idêntica à V34) ---
def exibir_glossario(termo_chave, lang_key):
    T = TEXT_DICT[lang_key]
    st.subheader(T['step4_glossary_header'])
    
    if termo_chave == "fickian":
        st.info(T['step4_glossary_fickian'])
    elif termo_chave == "anomalous":
        st.info(T['step4_glossary_anomalous'])
    elif termo_chave == "case_ii":
        st.info(T['step4_glossary_case_ii'])
    elif termo_chave == "zero_order":
        st.info(T['step4_glossary_zero_order'])
    elif termo_chave == "first_order":
        st.info(T['step4_glossary_first_order'])
    elif termo_chave == "hixson":
        st.info(T['step4_glossary_hixson'])
    elif termo_chave == "complex":
        st.info(T['step4_glossary_complex'])
    else:
        st.info(T['step4_glossary_select'])

# --- Função Principal do App ---
def main():
    st.set_page_config(layout="wide")
    # --- Seletor de Idioma ---
    if 'lang' not in st.session_state:
        st.session_state.lang = 'pt'
    lang_choice = st.sidebar.selectbox("Language / Idioma", ["Português", "English"], index=0 if st.session_state.lang == 'pt' else 1)
    st.session_state.lang = 'pt' if lang_choice == "Português" else 'en'
    T = TEXT_DICT[st.session_state.lang]

    st.title(T['app_title'])

    # --- Inicialização do Estado da Sessão ---
    if 'df_long_processado' not in st.session_state:
        st.session_state.df_long_processado = None
    if 'df_agregado' not in st.session_state:
        st.session_state.df_agregado = None
    if 'config' not in st.session_state:
        st.session_state.config = {}

    # --- NAVEGAÇÃO NA BARRA LATERAL (Esquerda) ---
    st.sidebar.title(T['sidebar_nav'])
    pagina = st.sidebar.radio(T['sidebar_nav'],
                            [T['nav_step1'],
                             T['nav_step2'],
                             T['nav_step3'],
                             T['nav_step4'],
                             T['nav_step5'],
                             T['nav_step6']]) 
    st.sidebar.markdown("---")
    st.sidebar.info(T['sidebar_info'])

    # --- ETAPA 1 (Idêntica à V34) ---
    if pagina == T['nav_step1']:
        st.header(T['step1_header'])
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(T['step1_subheader_units'])
            unidade_vol = st.selectbox(T['step1_label_vol_unit'], ["mL", "L", "µL"], 0)
            unidade_conc = st.selectbox(T['step1_label_conc_unit'], ["mg/mL", "µg/mL", "g/L", "mmol/L", "µmol/L"], 0)
            unidade_massa = unidade_conc.split('/')[0]
            st.markdown("---")
            st.subheader(T['step1_subheader_cell'])
            vol_celula = st.number_input(f"{T['step1_label_vol_receptor']} ({unidade_vol})", min_value=0.01, value=12.0, format="%.4f", step=0.1)
            vol_amostra = st.number_input(f"{T['step1_label_vol_sample']} ({unidade_vol})", min_value=0.0, value=1.0, format="%.4f", step=0.1)
        with col2:
            st.subheader(T['step1_subheader_calib'])
            st.markdown(f"`Área = a * Conc. ({unidade_conc}) + b`")
            cal_a = st.number_input(T['step1_label_calib_a'], value=10000.0, format="%.4f", step=1.0)
            cal_b = st.number_input(T['step1_label_calib_b'], value=0.0, format="%.4f", step=0.01, help=T['step1_help_calib_b'])
            st.markdown("---")
            st.subheader(T['step1_subheader_time'])
            unidade_tempo = st.selectbox(T['step1_label_time_unit'], [T['step1_time_hours'], T['step1_time_minutes']], 0)
        
        st.markdown("---")
        st.subheader(T['step1_subheader_upload'])
        with st.expander(T['step1_expander_model']):
             st.markdown("""
            | Amostra_Nome | Grupo | 0 | 0.5 | 1 | 2 |
            | :--- | :--- | :-: | :-: | :-: | :-: |
            | F1_Rep1 | F1 | 0 | 15023 | 28904 | 45002 |
            | F1_Rep2 | F1 | 0 | 14990 | 29100 | 44888 |
            | F2_Rep1 | F2 | 0 | 5000 | 9500 | 15000 |
            """)
        
        uploaded_file = st.file_uploader(T['step1_uploader_label'], type=["csv", "xlsx", "txt"])
        
        sep = ','
        decimal = '.'
        if uploaded_file and (uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt')):
            st.info(T['step1_csv_options'])
            col_parse1, col_parse2 = st.columns(2)
            with col_parse1:
                sep = st.selectbox(T['step1_csv_sep'], [T['step1_csv_sep_semi'], T['step1_csv_sep_comma'], T['step1_csv_sep_tab']], 0).split(" ")[0].replace("Tab","\t")
            with col_parse2:
                decimal = st.selectbox(T['step1_csv_dec'], [T['step1_csv_dec_comma'], T['step1_csv_dec_point']], 0).split(" ")[0]

        if uploaded_file is not None:
            if cal_a == 0:
                st.error(T['step1_error_calib_a'])
                return
            try:
                if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt'):
                    df_wide = load_data(uploaded_file, sep=sep, decimal=decimal)
                else:
                    df_wide = pd.read_excel(uploaded_file)
                
                if df_wide is None: 
                    st.error("Não foi possível ler o arquivo. Verifique o formato e a codificação.")
                    return

                st.write(T['step1_upload_preview'], df_wide.head())
                
                col_amostra_nome = df_wide.columns[0]
                col_grupo = df_wide.columns[1]
                cols_tempo = df_wide.columns[2:]
                
                st.markdown("---")
                st.subheader(T['step1_subheader_dose'])
                unique_groups = df_wide[col_grupo].unique()
                doses_dict = {}
                has_any_dose = False 

                for grupo in unique_groups:
                    st.markdown(f"**{T['step1_dose_group']} {grupo}**")
                    c1, c2 = st.columns(2)
                    conc_form = c1.number_input(f"{T['step1_dose_conc']} ({unidade_conc})", key=f"conc_{grupo}", min_value=0.0, value=10.0, format="%.4f", step=0.1)
                    vol_form = c2.number_input(f"{T['step1_dose_vol']} ({unidade_vol})", key=f"vol_{grupo}", min_value=0.0, value=1.0, format="%.4f", step=0.1)
                    dose_total_grupo = conc_form * vol_form
                    doses_dict[grupo] = {'dose_total': dose_total_grupo}
                    if dose_total_grupo > 0: has_any_dose = True

                st.markdown("---")
                if st.button(T['step1_button_process'], type="primary"):
                    with st.spinner(T['step1_spinner_process']):
                        df_long = df_wide.melt(id_vars=[col_amostra_nome, col_grupo], value_vars=cols_tempo, var_name='Tempo', value_name='Area')
                        df_long['Tempo'] = pd.to_numeric(df_long['Tempo'], errors='coerce')
                        df_long['Area'] = pd.to_numeric(df_long['Area'], errors='coerce')
                        df_long = df_long.dropna(subset=['Tempo', 'Area'], how='any')
                        
                        if unidade_tempo == T['step1_time_minutes']:
                            df_long['Tempo'] = df_long['Tempo'] / 60
                        
                        col_conc_nome = f"{T['step1_col_conc_name']} ({unidade_conc})"
                        col_q_acumulada_nome = f"{T['step1_col_q_name']} ({unidade_massa})"
                        col_percent_nome = T['step1_col_pct_name']

                        # --- CORREÇÃO V40: Removendo include_groups=False para evitar TypeError em Pandas antigos.
                        # O FutureWarning pode reaparecer em Pandas mais novos, mas o TypeError é prioridade.
                        df_long_processado = df_long.groupby(col_amostra_nome).apply(
                            calcular_liberacao_replica_v9,
                            col_grupo=col_grupo, vol_celula=vol_celula,
                            vol_amostra=vol_amostra, cal_a=cal_a, cal_b=cal_b,
                            doses_dict=doses_dict, col_conc=col_conc_nome,
                            col_q_acumulada=col_q_acumulada_nome,
                            col_percent=col_percent_nome
                        )
                        df_long_processado = df_long_processado.reset_index(drop=True)
                        
                        df_agregado = df_long_processado.groupby([col_grupo, 'Tempo']).agg(
                            Média_Q_Acumulada=(col_q_acumulada_nome, 'mean'),
                            SD_Q_Acumulada=(col_q_acumulada_nome, 'std'),
                            Média_Percent=(col_percent_nome, 'mean'),
                            SD_Percent=(col_percent_nome, 'std')
                        ).reset_index()
                        
                        df_agregado = df_agregado.fillna(0)
                        st.session_state.df_long_processado = df_long_processado
                        st.session_state.df_agregado = df_agregado
                        
                        config_dict = {
                            'unidade_massa': unidade_massa, 'has_dose_info': has_any_dose,
                            'col_q_acumulada': col_q_acumulada_nome, 'col_percent': col_percent_nome,
                            'col_conc': col_conc_nome, 'col_grupo': col_grupo,
                            'vol_celula': vol_celula,
                            'vol_amostra': vol_amostra,
                            'unidade_vol': unidade_vol,
                        }
                        
                        if has_any_dose:
                            config_dict['y_label'] = T['step1_y_label_pct']
                            config_dict['y_axis_mean'] = 'Média_Percent'
                            config_dict['y_axis_sd'] = 'SD_Percent'
                        else:
                            config_dict['y_label'] = f"{T['step1_y_label_q']} ({unidade_massa})"
                            config_dict['y_axis_mean'] = 'Média_Q_Acumulada'
                            config_dict['y_axis_sd'] = 'SD_Q_Acumulada'
                        
                        st.session_state.config = config_dict
                        st.success(T['step1_success_process'])
            except Exception as e:
                st.error(f"{T['step1_error_process']} {e}")
                st.exception(e)

    # --- ETAPA 2 (Idêntica à V34) ---
    elif pagina == T['nav_step2']:
        st.header(T['step2_header'])
        if st.session_state.df_long_processado is None:
            st.warning(T['step2_warning_process'])
            return
        st.info(T['step2_info'])
        st.subheader(T['step2_subheader_raw'])
        st.dataframe(st.session_state.df_long_processado)
        st.subheader(T['step2_subheader_agg'])
        st.dataframe(st.session_state.df_agregado)

    # --- ETAPA 3 (Gráficos - Idêntica à V34) ---
    elif pagina == T['nav_step3']:
        st.header(T['step3_header'])
        
        if st.session_state.df_agregado is None:
            st.warning(T['step2_warning_process'])
            return

        df_agg = st.session_state.df_agregado
        config = st.session_state.config
        col_grupo = config['col_grupo']
        grupos_disponiveis = df_agg[col_grupo].unique()
        
        y_axis_mean = config['y_axis_mean']
        y_axis_sd = config['y_axis_sd']
        y_label = config['y_label']

        st.info(T['step3_info_export'])

        st.subheader(T['step3_subheader_individual'])
        grupo_selecionado = st.selectbox(T['step3_selectbox_group'], grupos_disponiveis)
        
        if grupo_selecionado:
            df_grupo_agg = df_agg[df_agg[col_grupo] == grupo_selecionado].copy()
            fig_individual = go.Figure()
            
            fig_individual.add_trace(go.Scatter(
                x=df_grupo_agg['Tempo'], y=df_grupo_agg[y_axis_mean],
                mode='lines+markers', name=T['step3_legend_individual'],
                line=dict(color='blue', width=2), marker=dict(size=8),
                error_y=dict(
                    type='data', array=df_grupo_agg[y_axis_sd],
                    visible=True, thickness=1.5, width=3
                )
            ))
            
            fig_individual.update_layout(
                title=f"{T['step3_title_individual']} {grupo_selecionado}",
                xaxis_title=T['step3_xaxis_label'], yaxis_title=y_label, 
                template="plotly_white", 
                width=500, height=600, 
                xaxis_mirror=True, yaxis_mirror=True, 
                xaxis_linewidth=1, yaxis_linewidth=1,
                xaxis_linecolor='black', yaxis_linecolor='black',
                legend=dict(
                    orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5
                ),
                margin=dict(l=50, r=50, t=50, b=150) 
            )
            st.plotly_chart(fig_individual, use_container_width=False)

        st.subheader(T['step3_subheader_compare'])
        
        color_map = {}
        default_colors = px.colors.qualitative.Plotly
        with st.expander(T['step3_color_picker_label']):
            for i, grupo in enumerate(grupos_disponiveis):
                default_color = default_colors[i % len(default_colors)]
                color_map[grupo] = st.color_picker(f"{T['step3_color_picker_group']} {grupo}", value=default_color, key=f"color_{grupo}")
        
        fig_comparativo = go.Figure()
        
        for i, grupo in enumerate(grupos_disponiveis):
            df_grupo = df_agg[df_agg[col_grupo] == grupo]
            cor_hex = color_map.get(grupo)
            
            fig_comparativo.add_trace(go.Scatter(
                x=df_grupo['Tempo'],
                y=df_grupo[y_axis_mean],
                mode='lines+markers',
                name=grupo,
                line=dict(color=cor_hex, width=2),
                marker=dict(size=8),
                error_y=dict(
                    type='data',
                    array=df_grupo[y_axis_sd],
                    visible=True,
                    thickness=1.5,
                    width=3
                )
            ))

        fig_comparativo.update_layout(
            title=T['step3_title_compare'],
            xaxis_title=T['step3_xaxis_label'], yaxis_title=y_label,
            template="plotly_white", 
            width=500, height=600, 
            xaxis_mirror=True, yaxis_mirror=True, 
            xaxis_linewidth=1, yaxis_linewidth=1,
            xaxis_linecolor='black', yaxis_linecolor='black',
            legend=dict(
                orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5
            ),
            margin=dict(l=50, r=50, t=50, b=150) 
        )
        st.plotly_chart(fig_comparativo, use_container_width=False)

    # --- ETAPA 4 (Modelagem - ATUALIZADA V40) ---
    elif pagina == T['nav_step4']:
        st.header(T['step4_header'])
        
        if st.session_state.df_agregado is None:
            st.warning(T['step2_warning_process'])
            return

        df_agg = st.session_state.df_agregado
        config = st.session_state.config
        col_grupo = config['col_grupo']
        grupos_disponiveis = df_agg[col_grupo].unique()
        
        st.subheader(T['step4_info_model'])
        
        grupo_selecionado_modelagem = st.selectbox(T['step4_selectbox_group'], grupos_disponiveis, key="model_select")
        
        if grupo_selecionado_modelagem:
            df_grupo_agg = df_agg[df_agg[col_grupo] == grupo_selecionado_modelagem].copy()
            y_axis_mean = config['y_axis_mean']
            df_model = df_grupo_agg[df_grupo_agg["Tempo"] > 0].copy()
            t_data = df_model["Tempo"]
            q_data = df_model[y_axis_mean] 

            if len(q_data) < 3:
                st.error(T['step4_error_points'])
                return

            # --- Chamando v11 e capturando mensagens ---
            df_resultados, mensagens_modelagem = rodar_modelagem_v11(t_data, q_data, df_model, y_axis_mean, config['has_dose_info'])

            # --- Exibir alertas de modelagem (usando st.warning) ---
            if mensagens_modelagem:
                for msg in mensagens_modelagem:
                    st.warning(msg) 

            if df_resultados.empty:
                st.error(T['step4_error_fit'])
                return
            
            df_resultados_display = df_resultados.copy()
            df_resultados_display.index = [T.get(f'model_{idx.lower().replace("-","_")}', idx) for idx in df_resultados_display.index]

            st.subheader(f"{T['step4_subheader_results']} {grupo_selecionado_modelagem}")
            st.dataframe(
                df_resultados_display.style
                .highlight_max(subset=["R2"], color="lightgreen", axis=0)
                # Formatando NaN R2 como '-'
                .format("{:.4f}", na_rep="-", subset=pd.IndexSlice[:, ['R2', 'kKP', 'n', 'k0', 'k1', 'kH', 'kHC', 'a', 'b']])
            )

            st.subheader(T['step4_subheader_interp'])
            
            melhor_modelo, r2_melhor, k_val, k_unit, n_val, interpretacao, glossario_key = interpretar_resultados(df_resultados, config['y_label'], st.session_state.lang)
            
            if r2_melhor >= 0.05: # Usar o mesmo limite de corte para consistência
                st.success(f"{T['step4_success_model']} **{melhor_modelo}** (R² = {r2_melhor:.4f})")
                
                param_str = f"**{T['step4_interp_const']}** {k_val:.4f} ({k_unit})"
                if pd.notna(n_val):
                    param_str += f" | **{T['step4_interp_exp']}** {n_val:.3f}"
                st.markdown(param_str)
                
                st.info(f"**{T['step4_interp_interp']}** {interpretacao}")
                
                exibir_glossario(glossario_key, st.session_state.lang)
            else:
                # --- MENSAGEM DE ERRO DETALHADA ---
                st.error(
                    f"{T['interp_fail']} "
                    f"**O R² de todos os modelos ajustados é considerado estatisticamente insignificante ($R^2 < 0.05$)** ou nulo. "
                    f"Isso ocorre porque a curva de liberação para o grupo **{grupo_selecionado_modelagem}** é muito rasa, "
                    f"plana ou possui poucos pontos, impedindo que os algoritmos de ajuste encontrem um padrão cinético válido."
                )


    # --- ETAPA 5 (Resumo - ATUALIZADA V40) ---
    elif pagina == T['nav_step5']:
        st.header(T['step5_header'])
        
        if st.session_state.df_agregado is None:
            st.warning(T['step2_warning_process'])
            return
            
        st.info(T['step5_info'])

        df_agg = st.session_state.df_agregado
        config = st.session_state.config
        col_grupo = config['col_grupo']
        grupos_disponiveis = df_agg[col_grupo].unique()
        
        resumo_list = []

        with st.spinner(T['step5_spinner']):
            for grupo in grupos_disponiveis:
                df_grupo_agg = df_agg[df_agg[col_grupo] == grupo].copy()
                y_axis_mean = config['y_axis_mean']
                df_model = df_grupo_agg[df_grupo_agg["Tempo"] > 0].copy()
                t_data = df_model["Tempo"]
                q_data = df_model[y_axis_mean] 

                if len(q_data) < 3:
                    resumo_list.append({
                        T['step5_col_group']: grupo, T['step5_col_model']: "N/A", T['step5_col_r2']: 0,
                        T['step5_col_k']: np.nan,
                        T['step5_col_k_unit']: "-",
                        T['step5_col_n']: np.nan,
                        T['step5_col_interp']: T['step5_error_data']
                    })
                    continue
                
                # --- Chamando v11 e ignorando mensagens ---
                df_resultados_grupo, _ = rodar_modelagem_v11(t_data, q_data, df_model, y_axis_mean, config['has_dose_info'])
                
                melhor_modelo, r2_melhor, k_val, k_unit, n_val, interpretacao, _ = interpretar_resultados(df_resultados_grupo, config['y_label'], st.session_state.lang)
                
                resumo_list.append({
                    T['step5_col_group']: grupo,
                    T['step5_col_model']: melhor_modelo,
                    # Se r2_melhor < 0.05, ele é 0.0 aqui, mas o formatador abaixo lida com NaN/zeros
                    T['step5_col_r2']: r2_melhor, 
                    T['step5_col_k']: k_val,
                    T['step5_col_k_unit']: k_unit,
                    T['step5_col_n']: n_val,
                    T['step5_col_interp']: interpretacao
                })
        
        df_resumo = pd.DataFrame(resumo_list).set_index(T['step5_col_group'])
        
        # Aplica a conversão de R2 baixo para NaN para o display final
        df_resumo[T['step5_col_r2']] = df_resumo[T['step5_col_r2']].apply(lambda x: np.nan if x is not None and x < 0.05 else x)
        
        st.dataframe(
            df_resumo.style.format({
                T['step5_col_r2']: "{:.4f}",
                T['step5_col_k']: "{:.4f}",
                T['step5_col_n']: "{:.3f}"
            }, na_rep="-"),
            use_container_width=True
        )

    # --- ETAPA 6 (Gerador de Prompt de IA - ATUALIZADA V40) ---
    elif pagina == T['nav_step6']:
        st.header(T['step6_header'])
        
        if st.session_state.df_agregado is None or st.session_state.config == {}:
            st.warning(T['step2_warning_process'])
            return
            
        st.info(T['step6_info'])
        
        st.subheader(T['step6_subheader_context'])
        
        col1, col2 = st.columns(2)
        with col1:
            drug_name = st.text_input(T['step6_label_drug'], "Ex: Curcumina")
            release_medium = st.text_input(T['step6_label_medium'], "Ex: Tampão fosfato (pH 7.4) + Tween 80 0.5%")
        with col2:
            system_name = st.text_input(T['step6_label_system'], "Ex: Nanopartículas de PLGA")
            membrane_type = st.text_input(T['step6_label_membrane'], "Ex: Acetato de celulose (0.45 µm)")
        
        objective = st.text_input(T['step6_label_objective'], "Ex: Comparar a liberação da curcumina livre vs. nanoencapsulada")
        
        if st.button(T['step6_button_generate'], type="primary"):
            with st.spinner(T['step5_spinner']):
                # 1. Recalcular o Resumo da Etapa 5 
                df_agg = st.session_state.df_agregado
                config = st.session_state.config
                col_grupo = config['col_grupo']
                grupos_disponiveis = df_agg[col_grupo].unique()
                resumo_list = []
                
                for grupo in grupos_disponiveis:
                    df_grupo_agg = df_agg[df_agg[col_grupo] == grupo].copy()
                    y_axis_mean = config['y_axis_mean']
                    df_model = df_grupo_agg[df_grupo_agg["Tempo"] > 0].copy()
                    t_data = df_model["Tempo"]
                    q_data = df_model[y_axis_mean]
                    if len(q_data) < 3: continue
                    
                    # --- Chamando v11 e ignorando mensagens ---
                    df_resultados_grupo, _ = rodar_modelagem_v11(t_data, q_data, df_model, y_axis_mean, config['has_dose_info'])
                    
                    melhor_modelo, r2_melhor, k_val, k_unit, n_val, interpretacao, _ = interpretar_resultados(df_resultados_grupo, config['y_label'], st.session_state.lang)
                    resumo_list.append({
                        T['step5_col_group']: grupo,
                        T['step5_col_model']: melhor_modelo,
                        T['step5_col_r2']: r2_melhor,
                        T['step5_col_k']: k_val,
                        T['step5_col_k_unit']: k_unit,
                        T['step5_col_n']: n_val,
                    })
                
                df_resumo = pd.DataFrame(resumo_list).set_index(T['step5_col_group'])
                
                # Aplica a conversão de R2 baixo para NaN para o output do prompt
                df_resumo[T['step5_col_r2']] = df_resumo[T['step5_col_r2']].apply(lambda x: np.nan if x is not None and x < 0.05 else x)
                
                # 2. Obter Dados de Liberação Final
                y_axis_label = config['y_axis_mean']
                y_unit = "(%)" if '%' in y_axis_label else f"({config.get('unidade_massa', 'unidade')})"
                max_time = df_agg['Tempo'].max()
                
                final_results_str = f"| {T['step5_col_group']} | Tempo Final (h) | Liberação Média {y_unit} |\n"
                final_results_str += f"| :--- | :--- | :--- |\n"
                for grupo in grupos_disponiveis:
                    val = df_agg.loc[(df_agg[col_grupo] == grupo) & (df_agg['Tempo'] == max_time), y_axis_label].values
                    if len(val) > 0:
                        final_results_str += f"| {grupo} | {max_time} | {val[0]:.2f} |\n"
                
                # 3. Montar o String do Prompt
                prompt = f"{T['prompt_title']}\n\n"
                prompt += f"{T['prompt_context_header']}\n"
                prompt += f"* **{T['prompt_context_drug']}:** {drug_name}\n"
                prompt += f"* **{T['prompt_context_system']}:** {system_name}\n"
                prompt += f"* **{T['prompt_context_objective']}:** {objective}\n\n"
                
                prompt += f"{T['prompt_methods_header']}\n"
                prompt += f"* **{T['prompt_methods_vol_receptor']}:** {config.get('vol_celula', 'N/A')} {config.get('unidade_vol', '')}\n"
                prompt += f"* **{T['prompt_methods_vol_sample']}:** {config.get('vol_amostra', 'N/A')} {config.get('unidade_vol', '')}\n"
                prompt += f"* **{T['prompt_methods_medium']}:** {release_medium}\n" 
                prompt += f"* **{T['prompt_methods_membrane']}:** {membrane_type}\n" 
                prompt += "\n"
                
                prompt += f"{T['prompt_results_header']}\n"
                prompt += final_results_str
                prompt += "\n"
                
                prompt += f"{T['prompt_kinetics_header']}\n"
                # Usa to_string para obter um formato de tabela limpo com '-' para NaN
                prompt += df_resumo.to_string(float_format="%.4f", na_rep='-') 
                prompt += "\n\n"
                
                prompt += f"{T['prompt_tasks_header']}\n"
                prompt += f"{T['prompt_task_1']}\n"
                prompt += f"{T['prompt_task_2']}\n"
                prompt += f"{T['prompt_task_3']}\n"
                
                st.text_area(T['step6_prompt_header'], value=prompt, height=400)
                st.success(T['step6_copy_success'])


# Ponto de entrada do script
if __name__ == "__main__":
    main()