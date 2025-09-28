import math
import numpy as np
import pandas as pd
from keras.models import load_model
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ModelPipeline:

    def predict(self, X: pd.DataFrame, model_path: str = 'model.keras') -> np.ndarray:

        model = self.__load_model(model_path)
        pred = model.predict(X)
        
        return self.__normalize_predict(pred)

    def structure(self, data: dict) -> pd.DataFrame:

        logger.info(f'Structuring data into DataFrame: {data}')
        df = pd.read_json(data['data'])
        logger.info(f'DataFrame shape: {df}')
        
        return df

    def preprocessing(self, df: pd.DataFrame):

        logger.info('Starting preprocessing')
        df['dividend_yield'] = df['dividend_yield']/100
        df['cagr_lucros_5_anos'] = df['cagr_lucros_5_anos']/100
        df['margem_liquida'] = df['margem_liquida']/100
        df['annual_dividend'] = df['close_price']*df['dividend_yield']
        df['dividend_yield_meta'] = 0.12
        df['graham_price'] = 22.5*df['vpa']*df['lpa']
        df['graham_price'] = df['graham_price'].apply(lambda x: math.sqrt(x) if x >= 0 else -(math.sqrt(abs(x))))
        df['bazin_price'] = df['annual_dividend']/df['dividend_yield_meta']+((df['annual_dividend']/df['dividend_yield_meta'])*df['cagr_lucros_5_anos'])
        df['bazin_price'] = df['bazin_price'].fillna(float(0))
        df['bazin_price'] = df['bazin_price'].apply(lambda x: float(0) if str(x) in ['-inf', 'inf'] else x)
        df['peter_lynch_price'] = ((df['cagr_lucros_5_anos']+df['dividend_yield'])/df['p/l'])*100
        df['peter_lynch_price'] = df['peter_lynch_price'].fillna(float(0))
        df['peter_lynch_price'] = df['peter_lynch_price'].apply(lambda x: float(0) if str(x) in ['-inf', 'inf'] else x)
        df['segmento_score'] = df['setor_atua'].apply(lambda x: 50  if x == 'Consumo Cíclico'                 else
                                                                100 if x == 'Financeiro e Outros'             else
                                                                50  if x == 'Bens Industriais'                else
                                                                100 if x == 'Utilidade Pública'               else
                                                                75 if x == 'Materiais Básicos'                else
                                                                50 if x == 'Consumo não Cíclico'              else
                                                                75 if x == 'Saúde'                            else
                                                                25  if x == 'Tecnologia da Informação'        else
                                                                100 if x == 'Petróleo. Gás e Biocombustíveis' else
                                                                25  if x == 'Telecomunicações'                else
                                                                0)
        perene = ['Bancos', 'Seguradoras', 'Corretoras de Seguros',
                'Energia Elétrica', 'Água e Saneamento', 'Gás', 
                'Alimentos', 'Carnes e Derivados', 'Produtos de Limpeza', 
                'Alimentos Diversos', 'Serv.Méd.Hospit..Análises e Diagnósticos', 
                'Equipamentos', 'Medicamentos e Outros Produtos', 
                'Exploração. Refino e Distribuição']

        df['perene'] = df['segmento_atua'].apply(lambda x: 1 if x in perene else 0)
        df['liquidez_diaria_quant'] = df['liquidez_diaria']/df['close_price']
        df['liquidez_diaria_quant'] = df['liquidez_diaria_quant'].fillna(0)
        df['peter_lynch_tick'] = df['peter_lynch_price'].apply(lambda x: 1 if (x >= 2) & (x != 0) else 0)
        df['liquidez_diaria_quant_tick'] = df['liquidez_diaria_quant'].apply(lambda x: 1 if (x >= 100000) & (x != 0) else 0)
        df['divida_liq/ebit_tick'] = df['divida_liq/ebit'].apply(lambda x: 1 if (x < 2) & (x != 0) else 0)
        df['liquidez_corrente_tick'] = df['liquidez_corrente'].apply(lambda x: 1 if (x > 1) & (x != 0) else 0)
        df['margem_liquida_tick'] = df['margem_liquida'].apply(lambda x: 1 if (x >= 0.15) & (x != 0) else 0)
        df['graham_tick'] = df['graham_price'] - df['close_price'] > df['graham_price'] * 0.15
        df['graham_tick'] = df['graham_tick'].apply(lambda x: 1 if (x == True) & (x != 0) else 0)
        df['bazin_tick'] = df['close_price'] < df['bazin_price']
        df['bazin_tick'] = df['bazin_tick'].apply(lambda x: 1 if x == True else 0)
        df.drop(['setor_atua', 'subsetor_atua', 'segmento_atua'], axis=1, inplace=True)
        
        return df

    def __load_model(self, model_path: str):
        
        logger.info(f'Loading model from {model_path}')
        return load_model(model_path)

    def __normalize_predict(self, y_pred: np.ndarray) -> np.ndarray:
        logger.info('Normalizing predictions')
        sr_pred = pd.DataFrame(y_pred)
        orig_min = 0.123301804
        orig_max = 0.9601589
        pred = sr_pred.apply(lambda x: ((x - orig_min) / (orig_max - orig_min)) * 1000)
        pred = pred[0].apply(lambda x: 1000 if x > 1000 else 0 if x < 0 else x)
        return pred
