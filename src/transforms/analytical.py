from scipy.signal import hilbert
from tftb.processing import WignerVilleDistribution

class AnalyticalAnalyser:
    def __init__(self, signal):
        self._signal = signal
    
    def hilbert_transform(self):
        signal_hilbert = hilbert(self._signal)
        return signal_hilbert
    
    def wigner_ville_distribution(self):
        analysis = WignerVilleDistribution(self._signal)
        analysis.run()
        return analysis