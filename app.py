import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 1. BASE DE DATOS CIENTÍFICA (EDICIÓN PESOS MXN)
# ==============================================================================

# --- A. MATRIZ DE COMPATIBILIDAD ---
COMPATIBILIDAD = {
    'Maíz':    {'Tropical': 5, 'Desértico': 2, 'Templado': 4, 'Mediterráneo': 3},
    'Trigo':   {'Tropical': 1, 'Desértico': 1, 'Templado': 5, 'Mediterráneo': 4},
    'Tomate':  {'Tropical': 2, 'Desértico': 1, 'Templado': 4, 'Mediterráneo': 5},
    'Soja':    {'Tropical': 3, 'Desértico': 2, 'Templado': 5, 'Mediterráneo': 3},
    'Girasol': {'Tropical': 3, 'Desértico': 4, 'Templado': 3, 'Mediterráneo': 5},
    'Arroz':   {'Tropical': 5, 'Desértico': 1, 'Templado': 2, 'Mediterráneo': 3},
    'Papa':    {'Tropical': 1, 'Desértico': 1, 'Templado': 5, 'Mediterráneo': 3}
}

# --- B. PARÁMETROS BIOFÍSICOS (Precios ajustados a PESOS) ---
ESPECIES = {
    'Maíz': {
        'T_BASE': 10, 'T_OPT': 26, 'T_CEILING': 34, 
        'GDD_MAX': 1600, 'H_MAX': 250, 'RAIZ_MAX': 1200, 
        'RUE': 1.6, 'k_ext': 0.65, 'SLA': 0.016, 
        'HI': 0.50, 'PRECIO': 6.50, 'N_REQ': 1.5, 'RESISTENCIA_HELADA': 0 # $6.50 MXN/kg
    },
    'Trigo': {
        'T_BASE': 0, 'T_OPT': 20, 'T_CEILING': 26,
        'GDD_MAX': 2000, 'H_MAX': 100, 'RAIZ_MAX': 900,
        'RUE': 1.2, 'k_ext': 0.50, 'SLA': 0.022, 
        'HI': 0.45, 'PRECIO': 7.20, 'N_REQ': 1.1, 'RESISTENCIA_HELADA': -2
    },
    'Soja': {
        'T_BASE': 10, 'T_OPT': 25, 'T_CEILING': 32,
        'GDD_MAX': 1350, 'H_MAX': 90, 'RAIZ_MAX': 800,
        'RUE': 1.1, 'k_ext': 0.75, 'SLA': 0.025,
        'HI': 0.35, 'PRECIO': 10.50, 'N_REQ': 0.5, 'RESISTENCIA_HELADA': -1
    },
    'Tomate': {
        'T_BASE': 10, 'T_OPT': 24, 'T_CEILING': 30,
        'GDD_MAX': 1500, 'H_MAX': 180, 'RAIZ_MAX': 600,
        'RUE': 1.4, 'k_ext': 0.70, 'SLA': 0.028,
        'HI': 0.60, 'PRECIO': 22.00, 'N_REQ': 1.8, 'RESISTENCIA_HELADA': 2
    },
    'Girasol': {
        'T_BASE': 7, 'T_OPT': 26, 'T_CEILING': 35,
        'GDD_MAX': 1450, 'H_MAX': 210, 'RAIZ_MAX': 1300,
        'RUE': 1.5, 'k_ext': 0.80, 'SLA': 0.020,
        'HI': 0.35, 'PRECIO': 9.00, 'N_REQ': 1.0, 'RESISTENCIA_HELADA': -1
    },
    'Arroz': {
        'T_BASE': 12, 'T_OPT': 30, 'T_CEILING': 38,
        'GDD_MAX': 1700, 'H_MAX': 110, 'RAIZ_MAX': 400,
        'RUE': 1.4, 'k_ext': 0.55, 'SLA': 0.018,
        'HI': 0.50, 'PRECIO': 12.00, 'N_REQ': 1.2, 'RESISTENCIA_HELADA': 5
    },
    'Papa': {
        'T_BASE': 4, 'T_OPT': 17, 'T_CEILING': 24,
        'GDD_MAX': 1100, 'H_MAX': 60, 'RAIZ_MAX': 500,
        'RUE': 1.3, 'k_ext': 0.60, 'SLA': 0.020,
        'HI': 0.75, 'PRECIO': 14.50, 'N_REQ': 1.0, 'RESISTENCIA_HELADA': -1
    }
}

# --- C. PARÁMETROS CLIMÁTICOS ---
GEOGRAFIAS = {
    'Tropical': {
        'TMAX': 32, 'TMIN': 23, 'RH': 85, 'RAD': 21, 'Evap': 5.0, 
        'Plaga_Base': 0.30, 'Catastrofe': '🍄 Ataque de Hongos (Humedad)'
    },
    'Desértico': {
        'TMAX': 38, 'TMIN': 15, 'RH': 20, 'RAD': 28, 'Evap': 9.0, 
        'Plaga_Base': 0.10, 'Catastrofe': '🌪️ Tormenta de Arena (Abrasión)'
    },
    'Templado': {
        'TMAX': 26, 'TMIN': 12, 'RH': 65, 'RAD': 18, 'Evap': 3.5, 
        'Plaga_Base': 0.15, 'Catastrofe': '❄️ Helada Tardía'
    },
    'Mediterráneo': {
        'TMAX': 29, 'TMIN': 16, 'RH': 50, 'RAD': 25, 'Evap': 6.0, 
        'Plaga_Base': 0.15, 'Catastrofe': '🔥 Ola de Calor (Golpe de sol)'
    }
}

# --- D. TIPOS DE SUELO ---
SUELOS = {
    'Arenoso':   {'CC': 100, 'PMP': 10, 'Desc': 'Drena rápido. Riego frecuente.'},
    'Franco':    {'CC': 220, 'PMP': 60, 'Desc': 'Equilibrio ideal.'},
    'Arcilloso': {'CC': 300, 'PMP': 120, 'Desc': 'Retiene mucho. Difícil manejo.'}
}

# --- E. COSTOS (Ajustados a PESOS MXN) ---
COSTO_RIEGO = 250         # Antes 15 USD -> Ahora 250 MXN
COSTO_FERTILIZANTE = 500  # Antes 30 USD -> Ahora 500 MXN
COSTO_FITOSANITARIO = 800 # Antes 40 USD -> Ahora 800 MXN
RIEGO_MM = 30
DOSIS_NITROGENO = 40

# ==============================================================================
# 2. MOTOR DE SIMULACIÓN (CLASE CULTIVO)
# ==============================================================================
class Cultivo:
    def __init__(self, nombre_especie, nombre_geo, tipo_suelo_key):
        self.nombre_especie = nombre_especie
        self.cfg_esp = ESPECIES[nombre_especie]
        self.cfg_geo = GEOGRAFIAS[nombre_geo]
        
        # Estrellas y Adaptación
        self.stars = COMPATIBILIDAD.get(nombre_especie, {}).get(nombre_geo, 3)
        self.factor_adaptacion = 0.5 + (self.stars * 0.11) 
        
        # Suelo
        self.perfil_suelo = SUELOS[tipo_suelo_key]
        self.capacidad_campo = self.perfil_suelo['CC']
        self.pmp = self.perfil_suelo['PMP']
        
        # Estado Inicial
        self.agua_suelo = self.pmp + ((self.capacidad_campo - self.pmp) * 0.7)
        self.nitrogeno = 100.0
        self.dia = 0
        self.biomasa = 1.5
        self.gdd_acumulados = 0.0
        self.lai = 0.005
        self.altura = 0.0
        self.raiz_prof = 50.0
        self.dias_proteccion = 0
        
        self.hi_actual = self.cfg_esp['HI']
        self.costos = 0
        self.estado_sim = 'PLAY'
        self.logs = [f"🌱 Inicio: {nombre_especie}. Suelo: {tipo_suelo_key}. Adaptabilidad: {self.stars}★"]
        self.historial = []
        self.cause_of_death = ""
        self.pronostico_temp = 0

    def _calcular_vpd(self, t, rh):
        es = 0.6108 * np.exp((17.27 * t) / (t + 237.3))
        ea = es * (rh / 100.0)
        return max(0.1, es - ea)

    def simular_dia(self):
        if self.estado_sim != 'PLAY': return
        self.dia += 1

        # 1. CLIMA
        t_prom = np.random.normal((self.cfg_geo['TMAX'] + self.cfg_geo['TMIN'])/2, 2.5)
        rh_hoy = np.max([10, np.random.normal(self.cfg_geo['RH'], 8)])
        rad_hoy = np.max([5, np.random.normal(self.cfg_geo['RAD'], 4)])
        vpd = self._calcular_vpd(t_prom, rh_hoy)
        self.pronostico_temp = t_prom + np.random.normal(0, 1.5)

        # 2. HELADAS
        t_min_est = t_prom - 6
        if t_min_est < 0:
            umbral = self.cfg_esp['RESISTENCIA_HELADA']
            if t_min_est < umbral:
                daño_frio = 0.4
                self.biomasa *= (1 - daño_frio)
                self.lai *= (1 - daño_frio)
                self.logs.append(f"❄️ HELADA SEVERA ({t_min_est:.1f}°C). Daño estructural.")

        # 3. FENOLOGÍA
        t_base = self.cfg_esp['T_BASE']
        t_ceil = self.cfg_esp['T_CEILING']
        if t_prom < t_base: gdd_hoy = 0
        elif t_prom > t_ceil: 
            gdd_hoy = t_ceil - t_base
            self.biomasa *= 0.995 
        else: gdd_hoy = t_prom - t_base
        
        self.gdd_acumulados += gdd_hoy
        progreso = min(1.0, self.gdd_acumulados / self.cfg_esp['GDD_MAX'])

        # 4. RAÍCES
        if self.raiz_prof < self.cfg_esp['RAIZ_MAX']:
            self.raiz_prof += 12 * self.factor_adaptacion
        exploracion = min(1.0, self.raiz_prof / 1500)
        agua_disponible = self.agua_suelo * exploracion
        pmp_efectivo = self.pmp * exploracion
        
        # 5. LIMITANTES
        if agua_disponible > pmp_efectivo:
            cap_efectiva = self.capacidad_campo * exploracion
            f_agua = (agua_disponible - pmp_efectivo) / (cap_efectiva - pmp_efectivo)
            f_agua = max(0, min(1, f_agua))
        else: f_agua = 0.0
            
        f_nit = max(0, min(1, self.nitrogeno / 60.0))
        f_vpd = 1.0
        if vpd > 2.5: f_vpd = max(0.2, 1.0 - (vpd - 2.5)*0.5)

        limitante = min(f_agua, f_nit, f_vpd)

        # 6. CRECIMIENTO
        senescencia = 1.0 if progreso < 0.85 else (1.0 - (progreso-0.85)*4)
        peso_hojas = self.biomasa * 0.5 * max(0, senescencia)
        self.lai = peso_hojas * self.cfg_esp['SLA']
        
        k = self.cfg_esp['k_ext']
        luz_int = rad_hoy * (1 - np.exp(-k * self.lai))
        
        crecimiento = luz_int * self.cfg_esp['RUE'] * limitante * self.factor_adaptacion
        self.biomasa += crecimiento
        
        # 7. CONSUMO
        transpiracion = self.cfg_geo['Evap'] * (luz_int/25) * vpd * 0.9
        self.agua_suelo = max(self.pmp - 5, self.agua_suelo - transpiracion)
        self.nitrogeno -= crecimiento * self.cfg_esp['N_REQ'] * 0.01

        # 8. EVENTOS
        self._check_eventos(progreso)
        self.altura = self.cfg_esp['H_MAX'] * (1 / (1 + np.exp(-8 * (progreso - 0.4))))

        self.historial.append({
            "Día": self.dia, 
            "Biomasa": int(self.biomasa), 
            "Agua": int(self.agua_suelo),
            "LAI": round(self.lai, 2),
            "GDD": int(self.gdd_acumulados),
            "Estrés": round(1-f_agua, 2)
        })

        if progreso >= 1.0:
            self.estado_sim = 'HARVEST'
            self.logs.append(f"🚜 Madurez Fisiológica. ¡Cosecha lista!")

    def _check_eventos(self, progreso):
        if self.agua_suelo <= self.pmp:
            self.estado_sim = 'DEAD'
            self.cause_of_death = "Sequía"
            self.logs.append("💀 Cultivo muerto por falta de agua.")
            return

        factor_proteccion = 1.0
        if self.dias_proteccion > 0:
            self.dias_proteccion -= 1
            factor_proteccion = 0.05
            if self.dias_proteccion == 0: self.logs.append("⚠️ Protección expirada.")

        riesgo_map = {1: 3.0, 2: 2.0, 3: 1.0, 4: 0.8, 5: 0.5}
        prob_real = self.cfg_geo['Plaga_Base'] * riesgo_map[self.stars] * factor_proteccion
        
        if np.random.random() < prob_real:
            daño = 0.20 / max(1, self.stars * 0.8)
            self.biomasa *= (1 - daño)
            self.lai *= (1 - daño)
            catastrofe = self.cfg_geo['Catastrofe']
            self.logs.append(f"Día {self.dia}: ⚠️ {catastrofe}. Daño: -{int(daño*100)}%")

        if 0.6 < progreso < 0.8 and self.historial[-1]["Estrés"] > 0.6:
            self.hi_actual *= 0.97
            if self.dia % 3 == 0: self.logs.append("⚠️ Estrés en Floración.")

    def acciones_usuario(self, accion):
        if accion == 'riego':
            self.agua_suelo = min(self.capacidad_campo, self.agua_suelo + RIEGO_MM)
            self.costos += COSTO_RIEGO
            self.logs.append(f"Día {self.dia}: 💧 Riego aplicado")
        elif accion == 'fert':
            self.nitrogeno += DOSIS_NITROGENO
            self.costos += COSTO_FERTILIZANTE
            self.logs.append(f"Día {self.dia}: 🧪 Fertilización N")
        elif accion == 'fito':
            self.dias_proteccion += 12
            self.costos += COSTO_FITOSANITARIO
            self.logs.append(f"Día {self.dia}: 🛡️ Sanidad aplicada")

# ==============================================================================
# 3. INTERFAZ GRÁFICA (STREAMLIT)
# ==============================================================================
st.set_page_config(page_title="BioSim Ultimate v4.1 (MXN)", layout="wide", page_icon="🌾")

if 'cultivo' not in st.session_state:
    st.session_state.cultivo = None

with st.sidebar:
    st.title("🌾 BioSim Ultimate")
    if st.session_state.cultivo is None or st.button("🔄 Reiniciar"):
        geo = st.selectbox("Región", list(GEOGRAFIAS.keys()))
        esp = st.selectbox("Cultivo", list(ESPECIES.keys()))
        suelo_key = st.selectbox("Suelo", list(SUELOS.keys()))
        
        s_info = SUELOS[suelo_key]
        st.caption(f"CC: {s_info['CC']}mm | PMP: {s_info['PMP']}mm")
        
        stars = COMPATIBILIDAD.get(esp, {}).get(geo, 3)
        st.write(f"**Compatibilidad:** {'⭐'*stars}")
        if stars <= 2: st.error("⚠️ Alto Riesgo")
        elif stars == 5: st.success("🌟 Zona Óptima")
            
        if st.button("🌱 SEMBRAR", type="primary"):
            st.session_state.cultivo = Cultivo(esp, geo, suelo_key)
            st.rerun()
            
    if st.session_state.cultivo:
        c = st.session_state.cultivo
        st.divider()
        st.metric("Inversión", f"${c.costos}")
        st.subheader("🌤️ Pronóstico")
        t_pred = c.pronostico_temp if c.dia > 0 else (c.cfg_geo['TMAX']+c.cfg_geo['TMIN'])/2
        st.write(f"Esperado: **{int(t_pred)}°C**")
        if t_pred > c.cfg_esp['T_CEILING']: st.warning("🔥 Ola de Calor")

if st.session_state.cultivo:
    c = st.session_state.cultivo
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Día / GDD", f"{c.dia} / {int(c.gdd_acumulados)}")
    k2.metric("Biomasa", f"{int(c.biomasa)} g/m²")
    delta = int(c.agua_suelo - c.pmp)
    k3.metric("Agua Útil", f"{delta} mm", delta="Crítico" if delta < 20 else "Bien", delta_color="normal")
    prot = f"{c.dias_proteccion} días" if c.dias_proteccion > 0 else "Sin Escudo"
    k4.metric("Sanidad", prot, delta="Vulnerable" if c.dias_proteccion==0 else "Protegido")
    
    prog = min(1.0, c.gdd_acumulados / c.cfg_esp['GDD_MAX'])
    st.progress(prog, text=f"Fenología: {int(prog*100)}%")

    t1, t2, t3 = st.tabs(["🎮 Acciones", "📈 Datos", "📋 Reporte"])
    
    with t1:
        if c.estado_sim == 'PLAY':
            c1, c2, c3 = st.columns(3)
            c1.button(f"💧 Riego (${COSTO_RIEGO})", on_click=c.acciones_usuario, args=('riego',), use_container_width=True)
            c2.button(f"🧪 Nitrógeno (${COSTO_FERTILIZANTE})", on_click=c.acciones_usuario, args=('fert',), use_container_width=True)
            c3.button(f"🛡️ Sanidad (${COSTO_FITOSANITARIO})", on_click=c.acciones_usuario, args=('fito',), use_container_width=True)
            st.divider()
            col_av1, col_av2 = st.columns(2)
            if col_av1.button("📅 Pasar Día", use_container_width=True):
                c.simular_dia()
                st.rerun()
            if col_av2.button("⏩ Semana (x7)", use_container_width=True):
                for _ in range(7): c.simular_dia()
                st.rerun()
        elif c.estado_sim == 'HARVEST':
            st.balloons()
            st.success("¡Cosecha Lista!")
            if st.button("💰 VENDER"):
                c.estado_sim = 'FINISHED'
                st.rerun()
        elif c.estado_sim == 'DEAD':
            st.error(f"GAME OVER: {c.cause_of_death}")
            if st.button("Ver Análisis"):
                c.estado_sim = 'FINISHED'
                st.rerun()

    with t2:
        if c.historial:
            df = pd.DataFrame(c.historial).set_index("Día")
            st.line_chart(df[["Biomasa", "Agua"]])
            c1, c2 = st.columns(2)
            c1.area_chart(df["LAI"], color="#4CAF50")
            c2.line_chart(df["Estrés"])

    with t3:
        if c.estado_sim == 'FINISHED':
            factor_mercado = np.random.uniform(0.8, 1.3)
            precio_final = c.cfg_esp['PRECIO'] * factor_mercado
            rend = c.biomasa * 10 * c.hi_actual
            ingreso = rend * precio_final
            balance = ingreso - c.costos
            
            st.header("Balance Económico (MXN)")
            st.info(f"Mercado: Precio final al {int(factor_mercado*100)}% del promedio.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Rendimiento", f"{int(rend)} kg/ha")
            m2.metric("Ingresos", f"${int(ingreso)}")
            m3.metric("Neto", f"${int(balance)}", delta_color="normal")
            
            if balance > 0: st.success("🎉 ¡Ganancia!")
            else: st.error("📉 Pérdida.")
            
            st.subheader("Bitácora")
            for l in reversed(c.logs): st.text(f"> {l}")