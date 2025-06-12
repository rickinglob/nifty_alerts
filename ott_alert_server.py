import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
from datetime import datetime, timedelta
import logging
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ott_alerts.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_file='config.json') -> dict:
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        raise

# Initialize alerts history
alerts_history = []

def get_stock_data(symbol: str, period: str = "5d", interval: str = "30m") -> Optional[pd.DataFrame]:
    """
    Fetch stock data from Yahoo Finance with error handling
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            logger.warning(f"No data found for symbol: {symbol}")
            return None
            
        if len(data) < 20:
            logger.warning(f"Insufficient data for {symbol}: {len(data)} points")
            return None
            
        data = data.dropna()
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_var_function(src: pd.Series, length: int = 5) -> pd.Series:
    """
    Calculate VAR using Pine Script logic
    """
    try:
        valpha = 2 / (length + 1)
        vud1 = pd.Series(np.where(src > src.shift(1), src - src.shift(1), 0), index=src.index)
        vdd1 = pd.Series(np.where(src < src.shift(1), src.shift(1) - src, 0), index=src.index)
        
        vUD = vud1.rolling(window=9, min_periods=1).sum()
        vDD = vdd1.rolling(window=9, min_periods=1).sum()
        
        denominator = vUD + vDD
        vCMO = np.where(denominator != 0, (vUD - vDD) / denominator, 0)
        vCMO = pd.Series(vCMO, index=src.index).fillna(0)
        
        VAR = pd.Series(index=src.index, dtype=float)
        VAR.iloc[0] = src.iloc[0]
        
        for i in range(1, len(src)):
            alpha_factor = valpha * abs(vCMO.iloc[i])
            VAR.iloc[i] = (alpha_factor * src.iloc[i] + (1 - alpha_factor) * VAR.iloc[i-1])
        
        return VAR
    except Exception as e:
        logger.error(f"Error in VAR calculation: {str(e)}")
        return pd.Series(index=src.index, dtype=float)

def calculate_ott(data: pd.DataFrame, length: int = 5, percent: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate OTT indicator
    """
    try:
        if data is None or data.empty or len(data) < length:
            logger.warning("Insufficient data for OTT calculation")
            empty_series = pd.Series(index=data.index if data is not None else [], dtype=float)
            return empty_series, empty_series, empty_series
        
        src = data['Close'].copy()
        VAR = calculate_var_function(src, length)
        MAvg = VAR.copy()
        
        fark = MAvg * percent * 0.01
        longStop = MAvg - fark
        shortStop = MAvg + fark
        
        longStop_adj = pd.Series(index=data.index, dtype=float)
        shortStop_adj = pd.Series(index=data.index, dtype=float)
        dir_series = pd.Series(index=data.index, dtype=int)
        MT = pd.Series(index=data.index, dtype=float)
        OTT = pd.Series(index=data.index, dtype=float)
        
        longStop_adj.iloc[0] = longStop.iloc[0]
        shortStop_adj.iloc[0] = shortStop.iloc[0]
        dir_series.iloc[0] = 1
        
        for i in range(1, len(data)):
            if pd.notna(MAvg.iloc[i]) and pd.notna(longStop_adj.iloc[i-1]):
                if MAvg.iloc[i] > longStop_adj.iloc[i-1]:
                    longStop_adj.iloc[i] = max(longStop.iloc[i], longStop_adj.iloc[i-1])
                else:
                    longStop_adj.iloc[i] = longStop.iloc[i]
            else:
                longStop_adj.iloc[i] = longStop.iloc[i]
            
            if pd.notna(MAvg.iloc[i]) and pd.notna(shortStop_adj.iloc[i-1]):
                if MAvg.iloc[i] < shortStop_adj.iloc[i-1]:
                    shortStop_adj.iloc[i] = min(shortStop.iloc[i], shortStop_adj.iloc[i-1])
                else:
                    shortStop_adj.iloc[i] = shortStop.iloc[i]
            else:
                shortStop_adj.iloc[i] = shortStop.iloc[i]
            
            if (dir_series.iloc[i-1] == -1 and 
                pd.notna(MAvg.iloc[i]) and pd.notna(shortStop_adj.iloc[i-1]) and
                MAvg.iloc[i] > shortStop_adj.iloc[i-1]):
                dir_series.iloc[i] = 1
            elif (dir_series.iloc[i-1] == 1 and 
                  pd.notna(MAvg.iloc[i]) and pd.notna(longStop_adj.iloc[i-1]) and
                  MAvg.iloc[i] < longStop_adj.iloc[i-1]):
                dir_series.iloc[i] = -1
            else:
                dir_series.iloc[i] = dir_series.iloc[i-1]
        
        for i in range(len(data)):
            MT.iloc[i] = longStop_adj.iloc[i] if dir_series.iloc[i] == 1 else shortStop_adj.iloc[i]
            
            if pd.notna(MAvg.iloc[i]) and pd.notna(MT.iloc[i]):
                if MAvg.iloc[i] > MT.iloc[i]:
                    OTT.iloc[i] = MT.iloc[i] * (200 + percent) / 200
                else:
                    OTT.iloc[i] = MT.iloc[i] * (200 - percent) / 200
            else:
                OTT.iloc[i] = MT.iloc[i]
        
        return MAvg, OTT, dir_series
    
    except Exception as e:
        logger.error(f"Error in OTT calculation: {str(e)}")
        empty_series = pd.Series(index=data.index if data is not None else [], dtype=float)
        return empty_series, empty_series, empty_series

def detect_signals(MAvg: pd.Series, OTT: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Detect buy and sell signals
    """
    try:
        if MAvg.empty or OTT.empty:
            empty_signals = pd.Series([False] * len(MAvg), index=MAvg.index)
            return empty_signals, empty_signals
        
        OTT_shifted = OTT.shift(2)
        buy_signal = (MAvg > OTT_shifted) & (MAvg.shift(1) <= OTT_shifted.shift(1))
        sell_signal = (MAvg < OTT_shifted) & (MAvg.shift(1) >= OTT_shifted.shift(1))
        
        buy_signal = buy_signal.fillna(False)
        sell_signal = sell_signal.fillna(False)
        
        return buy_signal, sell_signal
    
    except Exception as e:
        logger.error(f"Error in signal detection: {str(e)}")
        empty_signals = pd.Series([False] * len(MAvg), index=MAvg.index)
        return empty_signals, empty_signals

def validate_email_settings(email_settings: dict) -> bool:
    """Validate email settings"""
    required_fields = ['email', 'password', 'recipient']
    return all(email_settings.get(field) for field in required_fields)

def send_email_alert(symbol: str, signal_type: str, price: float, email_settings: dict) -> bool:
    """
    Send email alert
    """
    try:
        if not validate_email_settings(email_settings):
            logger.warning("Email settings incomplete")
            return False
        
        msg = MIMEMultipart()
        msg['From'] = email_settings['email']
        msg['To'] = email_settings['recipient']
        msg['Subject'] = f"OTT Alert: {signal_type.upper()} Signal for {symbol}"
        
        body = f"""
        OTT Strategy Alert
        
        Symbol: {symbol}
        Signal: {signal_type.upper()}
        Price: ₹{price:.2f}
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        This is an automated alert from your OTT trading system.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(email_settings['smtp_server'], email_settings['smtp_port']) as server:
            server.starttls()
            server.login(email_settings['email'], email_settings['password'])
            text = msg.as_string()
            server.sendmail(email_settings['email'], email_settings['recipient'], text)
        
        logger.info(f"Email alert sent for {symbol} - {signal_type}")
        return True
        
    except Exception as e:
        logger.error(f"Email sending failed: {str(e)}")
        return False

def add_alert_to_history(symbol: str, signal_type: str, price: float):
    """Add alert to history with deduplication"""
    global alerts_history
    alert = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': symbol,
        'signal': signal_type,
        'price': price
    }
    
    current_time = datetime.now()
    for existing_alert in alerts_history[:5]:
        existing_time = datetime.strptime(existing_alert['timestamp'], '%Y-%m-%d %H:%M:%S')
        if (existing_alert['symbol'] == symbol and 
            existing_alert['signal'] == signal_type and
            (current_time - existing_time).total_seconds() < 300):
            return
    
    alerts_history.insert(0, alert)
    if len(alerts_history) > 100:
        alerts_history = alerts_history[:100]

def main():
    """Main monitoring loop"""
    watchlist =[
    "^NSEI",
    "^BSESN",
    "^NSEBANK",
    "AARTIIND.NS",
    "ABB.NS",
    "ABCAPITAL.NS",
    "ABFRL.NS",
    "ACC.NS",
    "ADANIENSOL.NS",
    "ADANIENT.NS",
    "ADANIGREEN.NS",
    "ADANIPORTS.NS",
    "ALKEM.NS",
    "AMBUJACEM.NS",
    "ANGELONE.NS",
    "APLAPOLLO.NS",
    "APOLLOHOSP.NS",
    "ASHOKLEY.NS",
    "ASIANPAINT.NS",
    "ASTRAL.NS",
    "ATGL.NS",
    "AUBANK.NS",
    "AUROPHARMA.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJAJFINSV.NS",
    "BAJFINANCE.NS",
    "BALKRISIND.NS",
    "BANDHANBNK.NS",
    "BANKBARODA.NS",
    "BANKINDIA.NS",
    "BDL.NS",
    "BEL.NS",
    "BHARATFORG.NS",
    "BHARTIARTL.NS",
    "BHEL.NS",
    "BIOCON.NS",
    "BLUESTARCO.NS",
    "BOSCHLTD.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "BSE.NS",
    "BSOFT.NS",
    "CAMS.NS",
    "CANBK.NS",
    "CDSL.NS",
    "CESC.NS",
    "CGPOWER.NS",
    "CHAMBLFERT.NS",
    "CHOLAFIN.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "COFORGE.NS",
    "COLPAL.NS",
    "CONCOR.NS",
    "CROMPTON.NS",
    "CUMMINSIND.NS",
    "CYIENT.NS",
    "DABUR.NS",
    "DALBHARAT.NS",
    "DELHIVERY.NS",
    "DIVISLAB.NS",
    "DIXON.NS",
    "DLF.NS",
    "DMART.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "ETERNAL.NS",
    "EXIDEIND.NS",
    "FEDERALBNK.NS",
    "FORTIS.NS",
    "GAIL.NS",
    "GLENMARK.NS",
    "GMRAIRPORT.NS",
    "GODREJCP.NS",
    "GODREJPROP.NS",
    "GRANULES.NS",
    "GRASIM.NS",
    "HAL.NS",
    "HAVELLS.NS",
    "HCLTECH.NS",
    "HDFCAMC.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HFCL.NS",
    "HINDALCO.NS",
    "HINDCOPPER.NS",
    "HINDPETRO.NS",
    "HINDUNILVR.NS",
    "HINDZINC.NS",
    "HUDCO.NS",
    "ICICIBANK.NS",
    "ICICIGI.NS",
    "ICICIPRULI.NS",
    "IDEA.NS",
    "IDFCFIRSTB.NS",
    "IEX.NS",
    "IGL.NS",
    "IIFL.NS",
    "INDHOTEL.NS",
    "INDIANB.NS",
    "INDIGO.NS",
    "INDUSINDBK.NS",
    "INDUSTOWER.NS",
    "INFY.NS",
    "INOXWIND.NS",
    "IOC.NS",
    "IRB.NS",
    "IRCTC.NS",
    "IREDA.NS",
    "IRFC.NS",
    "ITC.NS",
    "JINDALSTEL.NS",
    "JIOFIN.NS",
    "JSL.NS",
    "JSWENERGY.NS",
    "JSWSTEEL.NS",
    "JUBLFOOD.NS",
    "KALYANKJIL.NS",
    "KAYNES.NS",
    "KEI.NS",
    "KOTAKBANK.NS",
    "KPITTECH.NS",
    "LAURUSLABS.NS",
    "LICHSGFIN.NS",
    "LICI.NS",
    "LODHA.NS",
    "LT.NS",
    "LTF.NS",
    "LTIM.NS",
    "LUPIN.NS",
    "M&M.NS",
    "M&MFIN.NS",
    "MANAPPURAM.NS",
    "MANKIND.NS",
    "MARICO.NS",
    "MARUTI.NS",
    "MAXHEALTH.NS",
    "MAZDOCK.NS",
    "MCX.NS",
    "MFSL.NS",
    "MGL.NS",
    "MOTHERSON.NS",
    "MPHASIS.NS",
    "MUTHOOTFIN.NS",
    "NATIONALUM.NS",
    "NAUKRI.NS",
    "NBCC.NS",
    "NCC.NS",
    "NESTLEIND.NS",
    "NHPC.NS",
    "NMDC.NS",
    "NTPC.NS",
    "NYKAA.NS",
    "OBEROIRLTY.NS",
    "OFSS.NS",
    "OIL.NS",
    "ONGC.NS",
    "PAGEIND.NS",
    "PATANJALI.NS",
    "PAYTM.NS",
    "PEL.NS",
    "PERSISTENT.NS",
    "PETRONET.NS",
    "PFC.NS",
    "PHOENIXLTD.NS",
    "PIDILITIND.NS",
    "PIIND.NS",
    "PNB.NS",
    "PNBHOUSING.NS",
    "POLICYBZR.NS",
    "POLYCAB.NS",
    "POONAWALLA.NS",
    "POWERGRID.NS",
    "PPLPHARMA.NS",
    "PRESTIGE.NS",
    "RBLBANK.NS",
    "RECLTD.NS",
    "RELIANCE.NS",
    "RVNL.NS",
    "SAIL.NS",
    "SBICARD.NS",
    "SBILIFE.NS",
    "SBIN.NS",
    "SHREECEM.NS",
    "SHRIRAMFIN.NS",
    "SIEMENS.NS",
    "SJVN.NS",
    "SOLARINDS.NS",
    "SONACOMS.NS",
    "SRF.NS",
    "SUNPHARMA.NS",
    "SUPREMEIND.NS",
    "SYNGENE.NS",
    "TATACHEM.NS",
    "TATACOMM.NS",
    "TATACONSUM.NS",
    "TATAELXSI.NS",
    "TATAMOTORS.NS",
    "TATAPOWER.NS",
    "TATASTEEL.NS",
    "TATATECH.NS",
    "TCS.NS",
    "TECHM.NS",
    "TIINDIA.NS",
    "TITAGARH.NS",
    "TITAN.NS",
    "TORNTPHARM.NS",
    "TORNPOWER.NS",
    "TRENT.NS",
    "TVSMOTOR.NS",
    "ULTRACEMCO.NS",
    "UNIONBANK.NS",
    "UNITDSPR.NS",
    "UNOMINDA.NS",
    "UPL.NS",
    "VBL.NS",
    "VEDL.NS",
    "VOLTAS.NS",
    "WIPRO.NS",
    "YESBANK.NS",
    "ZYDUSLIFE.NS"
]
    email_settings = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "email": "rakesh.msr@gmail.com",
    "password": "oick mmdo veiv dujt",
    "recipient": "niftystockalert@gmail.com",
    "enabled": True
  }
    ott_period = 5
    ott_percent = 1.5
    scan_interval = 300
    
    logger.info("Starting OTT Alert Server")
    while True:
        logger.info(f"Starting new scan at {datetime.now()}")
        alerts_found = 0
        
        for symbol in watchlist:
            try:
                data = get_stock_data(symbol, period="5d", interval="30m")
                
                if data is not None and len(data) > 20:
                    MAvg, OTT, dir_series = calculate_ott(data, ott_period, ott_percent)
                    buy_signals, sell_signals = detect_signals(MAvg, OTT)
                    
                    recent_buy = buy_signals.tail(3).any()
                    recent_sell = sell_signals.tail(3).any()
                    current_price = data['Close'].iloc[-1]
                    
                    if recent_buy:
                        alerts_found += 1
                        logger.info(f"BUY Signal - {symbol} at ₹{current_price:.2f}")
                        add_alert_to_history(symbol, "BUY", current_price)
                        if email_settings.get('enabled', False):
                            send_email_alert(symbol, "BUY", current_price, email_settings)
                    
                    elif recent_sell:
                        alerts_found += 1
                        logger.info(f"SELL Signal - {symbol} at ₹{current_price:.2f}")
                        add_alert_to_history(symbol, "SELL", current_price)
                        if email_settings.get('enabled', False):
                            send_email_alert(symbol, "SELL", current_price, email_settings)
                    
                else:
                    logger.warning(f"Unable to fetch data for {symbol}")
                
                time.sleep(0.5)  # Prevent rate limiting
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
        
        logger.info(f"Scan completed. Found {alerts_found} alerts.")
        logger.info(f"Sleeping for {scan_interval} seconds...")
        time.sleep(scan_interval)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down OTT Alert Server")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")