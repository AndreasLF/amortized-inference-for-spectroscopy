import numpy as np
import matplotlib.pyplot as plt
# matplotlib style
plt.style.use('seaborn')


class pseudoVoigtSimulator:
    def __init__(self, wavenumbers):
        """Simulate a pseudo-Voigt spectrum
        
        Args:
            wavenumbers (np.array): Wavenumbers
            peaks (np.array): Peak centers
            gamma (np.array): Peak widths
            eta (np.array): Mixing parameters
            alpha (np.array): Peak heights
        """
        self.wavenumbers = wavenumbers

    def lorentzian(self, w, c, gamma, height_nomralize=True):
        """Lorentzian function
        Args:
            w (np.array): wavenumber
            c (float): Center of the peak
            gamma (float): Width of the peak
        Returns:
            np.array: The Lorentzian function
        """
        if height_nomralize:
            # Height normalized lorentzian
            Ls = (1 / np.pi * gamma) / ((w - c)**2 + gamma**2)
            # divide by the maximum value
            Ls = Ls / np.max(Ls)
            return Ls

        else: 
            return (1 / np.pi * gamma) / ((w - c)**2 + gamma**2)
            
    def gaussian(self, w, c, gamma, height_normalize=True):
        """Gaussian function
        
        Args:
            w (np.array): wavenumber
            c (float): Center of the peak
            gamma (float): Width of the peak

        Returns:
            np.array: The Gaussian function
        """
        if height_normalize:
            # Height normalized gaussian
            Gs = 1/(np.sqrt(2*np.pi)*gamma) * np.exp((-(w-c)**2)/(2*gamma**2))
            # divide by the maximum value
            Gs = Gs / np.max(Gs)
            return Gs
        else: 
            return 1/(np.sqrt(2*np.pi)*gamma) * np.exp((-(w-c)**2)/(2*gamma**2))

    def pseudo_voigt(self, W, c, gamma, eta, height_normalize=True):
        """Pseudo-Voigt function
        Args:
            W (np.array): Wavenumbers
            c (float): Center of the peak
            gamma (float): Width of the peak
            eta (float): Mixing parameter
        Returns:
            np.array: The pseudo-Voigt function
        """
        ws = np.arange(W)

        K = len(c)
        
        wavenumbers = np.tile(ws, (K, 1))
        cs = np.tile(c, (W, 1)).T
        gammas = np.tile(gamma, (W, 1)).T
        etas = np.tile(eta, (W, 1)).T
        
        vp = etas*self.lorentzian(wavenumbers, cs, gammas, height_normalize) + (1 - etas)*self.gaussian(wavenumbers, cs, gammas, height_normalize)

        return vp

    def gaussian_noise(self, K, n):
        """Get gaussian noise
        
        Args:
            K (int): Number of wavenumbers
            n (int): Number of noise vec
            
        Returns:
            np.array: Gaussian noise
        """
        gaussian_noise = np.random.normal(0, 0.1, K)
        # gaussian_noise = np.tile(gaussian_noise, (2, 1))
        return gaussian_noise

    def generate_full_spectrum(self, peaks, gamma, eta, alpha):
        """Generate full spectrum

        Args:
            peaks (np.array): Peak centers
            gamma (np.array): Peak widths
            eta (np.array): Mixing parameters
            alpha (np.array): Peak heights

        Returns:
            np.array: Full spectrum
        """

        alpha = np.tile(alpha, (self.wavenumbers, 1)).T
        Vp = self.pseudo_voigt(self.wavenumbers, peaks, gamma, eta)
        voigt_with_alpha = Vp * alpha 
        full_spectrum = np.sum(voigt_with_alpha, axis=0) + self.gaussian_noise(self.wavenumbers, 1)
        return full_spectrum

if __name__ == "__main__":
    c = np.array([250,350])
    gamma = np.array([20,20])
    eta = np.array([0.5,0.5])
    alpha = np.array([1,2])
    # Vp = pseudo_voigt(500, c, gamma, eta)
    ps = pseudoVoigtSimulator(500)
    fs = ps.generate_full_spectrum(c, gamma, eta, alpha)

    plt.plot(fs)
    plt.show()