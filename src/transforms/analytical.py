from scipy import signal
from tftb.processing import WignerVilleDistribution

class AnalyticalAnalyser:
    def __init__(self, signal):
        self._signal = signal
    
    def hilbert_transform(self):
        return signal.hilbert(self._signal)
    
    def wigner_ville_distribution(self):
        analysis = WignerVilleDistribution(self._signal)
        analysis.run()
        return analysis