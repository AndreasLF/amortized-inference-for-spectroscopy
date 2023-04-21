import numpy as np
import matplotlib.pyplot as plt
import torch
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

        K = 1 if len(c.shape) == 1 else c.shape[0] 

        
        wavenumbers = np.tile(ws, (K, 1))
        cs = np.tile(c, (W, 1)).T
        gammas = np.tile(gamma, (W, 1)).T
        etas = np.tile(eta, (W, 1)).T
        
        vp = etas*self.lorentzian(wavenumbers, cs, gammas, height_normalize) + (1 - etas)*self.gaussian(wavenumbers, cs, gammas, height_normalize)

        return vp

    def gaussian_noise(self, K, sigma):
        """Get gaussian noise
        
        Args:
            K (int): Number of wavenumbers
            n (int): Number of noise vec
            
        Returns:
            np.array: Gaussian noise
        """
        gaussian_noise = np.random.normal(0, sigma, K)
        # gaussian_noise = np.tile(gaussian_noise, (2, 1))
        return gaussian_noise

    def generate_full_spectrum(self, peaks, gamma, eta, alpha, sigma = 0.5):
        """Generate full spectrum

        Args:
            peaks (np.array): Peak centers
            gamma (np.array): Peak widths
            eta (np.array): Mixing parameters
            alpha (np.array): Peak heights

        Returns:
            np.array: Full spectrum
        """

        # sigma = np.sum(alpha)*noise_to_signal_ratio

        alpha = np.tile(alpha, (self.wavenumbers, 1)).T
        Vp = self.pseudo_voigt(self.wavenumbers, peaks, gamma, eta)
        voigt_with_alpha = Vp * alpha 
        full_spectrum = np.sum(voigt_with_alpha, axis=0) + self.gaussian_noise(self.wavenumbers, sigma)
        return full_spectrum



    def generate_random_spectra(self, amount, peaks = None, gamma = None, eta = None, alpha = None):
        """Generate random spectra
        
        Args:
            amount (int): Number of spectra
            wavenumbers (np.array): Wavenumbers
            peaks (np.array): Peak centers
            gamma (np.array): Peak widths
            eta (np.array): Mixing parameters
            alpha (np.array): Peak heights
        """

        if peaks is None:
            peaks = np.array([250,350])
        if gamma is None:
            gamma = np.array([20,20])
        if eta is None:
            eta = np.array([0.5,0.5])
        if alpha is None:
            alpha = np.array([1,1])
        
        
        random_spectra = np.zeros((amount, self.wavenumbers))
        for i in range(amount):
            ps = self.generate_full_spectrum(peaks, gamma, eta, alpha)
            random_spectra[i] = ps


        # peaks = np.array([250])
        # gamma = np.array([20])
        # eta = np.array([0.5])
        # # Randomize alpha  with shape (amount, len(peaks))
        # alpha = np.random.uniform(0.5, 10, (amount, len(peaks)))

        # # # Sample alpha from a normal distribution
        # # alpha = np.random.normal(5, 2, (amount, len(peaks)))


        # # print(alpha.shape)

        # random_spectra = np.zeros((amount, self.wavenumbers))
        # for i in range(amount):
        #     ps = self.generate_full_spectrum(peaks, gamma, eta, alpha[i])
        #     random_spectra[i] = ps

        # return random_spectra
    

    
        # peaks = np.array([250])
        gamma = np.array([20])
        eta = np.array([0.5])
        alpha = np.array([1])

        #  Randomize peaks with shape (amount, len(peaks))
        peaks = np.random.uniform(50, 450, (amount, len(alpha)))

        # sample peaks from a normal distribution
        # peaks = np.random.normal(250, 100, (amount, len(alpha)))

        # print(alpha.shape)

        random_spectra = np.zeros((amount, self.wavenumbers))
        for i in range(amount):
            ps = self.generate_full_spectrum(peaks[i], gamma, eta, alpha)
            random_spectra[i] = ps

        return random_spectra

class pseudoVoigtSimulatorTorch:
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

    def lorentzian(self, w, c, gamma):
        """Lorentzian function
        Args:
            w (np.array): wavenumber
            c (float): Center of the peak
            gamma (float): Width of the peak
        Returns:
            np.array: The Lorentzian function
        """
        # if height_nomralize:
        #     # Height normalized lorentzian
        #     Ls = (1 / np.pi * gamma) / ((w - c)**2 + gamma**2)
        #     # divide by the maximum value
        #     Ls = Ls / torch.max(Ls)
        #     return Ls
        # 
        # else: 
        return (1 / np.pi * gamma) / ((w - c)**2 + gamma**2)
            
    def gaussian(self, w, c, gamma):
        """Gaussian function
        
        Args:
            w (np.array): wavenumber
            c (float): Center of the peak
            gamma (float): Width of the peak

        Returns:
            np.array: The Gaussian function
        """
        # if height_normalize:
        #     # Height normalized gaussian
        #     Gs = 1/(np.sqrt(2*torch.pi)*gamma) * torch.exp((-(w-c)**2)/(2*gamma**2))
        #     # divide by the maximum value
        #     Gs = Gs / torch.max(Gs)
        #     return Gs
        # else: 
        return 1/(np.sqrt(2*np.pi)*gamma) * torch.exp((-(w-c)**2)/(2*gamma**2))

    def pseudo_voigt(self, W, c, gamma, eta, height_normalize=False, wavenumber_normalize=False, batch_size=1):
        """Pseudo-Voigt function
        Args:
            W (np.array): Wavenumbers
            c (float): Center of the peak
            gamma (float): Width of the peak
            eta (float): Mixing parameter
        Returns:
            np.array: The pseudo-Voigt function
        """
        # ws = torch.arange(W)
        if wavenumber_normalize:
            ws = torch.linspace(0, 1, W)
            gamma = gamma / W
            c = c / W
        else: 
            ws = torch.arange(W)

        K = 1 if not c.shape else c.shape[0] 

        wavenumbers = torch.tile(ws, (K, 1))
        cs = torch.tile(c, (W, 1)).T
        gammas = torch.tile(gamma, (W, 1)).T
        etas = torch.tile(eta, (W, 1)).T
        
        vp = etas*self.lorentzian(wavenumbers, cs, gammas) + (1 - etas)*self.gaussian(wavenumbers, cs, gammas)
        
        if height_normalize:
            vp  = vp / torch.max(vp)
        return vp

    def gaussian_noise(self, K, sigma):
        """Get gaussian noise
        
        Args:
            K (int): Number of wavenumbers
            n (int): Number of noise vec
            
        Returns:
            np.array: Gaussian noise
        """
        # Make gaussian noise with torch 
        if sigma == 0:
            return torch.zeros(K)
        else:
            gaussian_noise = torch.distributions.normal.Normal(0, sigma).sample((K,))
            return gaussian_noise

    def generate_full_spectrum(self, peaks, gamma, eta, alpha, sigma = 0.5, height_normalize=False, wavenumber_normalize=False):
        """Generate full spectrum

        Args:
            peaks (np.array): Peak centers
            gamma (np.array): Peak widths
            eta (np.array): Mixing parameters
            alpha (np.array): Peak heights

        Returns:
            np.array: Full spectrum
        """

        # sigma = torch.sum(alpha)*noise_to_signal_ratio

        alpha = torch.tile(alpha, (self.wavenumbers, 1)).T
        Vp = self.pseudo_voigt(self.wavenumbers, peaks, gamma, eta, height_normalize=height_normalize, wavenumber_normalize=wavenumber_normalize)
        voigt_with_alpha = Vp * alpha 
        full_spectrum = torch.sum(voigt_with_alpha, axis=0) + self.gaussian_noise(self.wavenumbers, sigma)
        return full_spectrum
    

    def decoder(self, peaks, gamma, eta, alpha, height_normalize=False, wavenumber_normalize=False, batch_size = 1):
        W = self.wavenumbers            
        if wavenumber_normalize:
            ws = torch.linspace(0, 1, W)
            gamma = gamma / W
            # peaks = peaks / W
        else: 
            ws = torch.linspace(0, W, W)

        K = 1 if not peaks.shape else peaks.shape[0] 

        wavenumbers = torch.tile(ws, (K, batch_size)) 
        cs = torch.tile(peaks, (W, batch_size)).T if peaks.shape == torch.Size([1]) else peaks.repeat(1, W)
        # cs = peaks.repeat(1, W)
        gammas = torch.tile(gamma, (W, batch_size)).T
        etas = torch.tile(eta, (W, batch_size)).T
        alphas = torch.tile(alpha, (W, batch_size)).T if alpha.shape == torch.Size([1]) else alpha.repeat(1, W)


        ll = self.lorentzian(ws, cs, gammas)
        gs = self.gaussian(ws, cs, gammas)
        pv = etas*ll + (1 - etas)*gs
        if height_normalize:
            pv = pv/pv.max(1, keepdim=True)[0]
        pv = alphas * pv
        return pv

        


    def generator(self, K, peaks = torch.tensor([250]), gamma = torch.tensor([20]), 
                  eta = torch.tensor([0.5]), alpha = torch.tensor([5]), sigma = 0.5):
        """Generator for random spectra

        Args:
            K (int): Number of spectra
            peaks (tuple, optional): Peak centers. Defaults to torch.tensor([250]). If tuple, random numbers between the two values will be picked.
            gamma (tuple, optional): Peak widths. Defaults to torch.tensor([20]). If tuple, random numbers between the two values will be picked.
            eta (tuple, optional): Mixing parameters. Defaults to torch.tensor([0.5]). If tuple, random numbers between the two values will be picked.
            alpha (tuple, optional): Peak heights. Defaults to torch.tensor([5]). If tuple, random numbers between the two values will be picked.
            sigma (float, optional): Noise to signal ratio. Defaults to 0.5.

        Yields:
            torch.tensor: Full spectrum
        """
        c = peaks
        g = gamma
        e = eta
        a = alpha
        
        while True:
            # Randomize peaks
            if isinstance(peaks, tuple):
                # pick random number between 1 and 10
                c = torch.rand(K,) * (peaks[1]-peaks[0]) + peaks[0]

            if isinstance(gamma, tuple):
                # pick random number between 1 and 10
                g = torch.rand(K,) * (gamma[1]-gamma[0]) + gamma[0]
            
            if isinstance(eta, tuple):
                # pick random number between 1 and 10
                e = torch.rand(K,) * (eta[1]-eta[0]) + eta[0]
            
            # if alpha is tuple then randomize
            if isinstance(alpha, tuple):
                # pick random number between 1 and 10
                a = torch.rand(K,) * (alpha[1]-alpha[0]) + alpha[0]
        
            parameters = torch.stack((c, g, e, a))

            yield self.generate_full_spectrum(c, g, e, a, sigma, height_normalize=True, wavenumber_normalize=True), parameters


    def generate_random_spectra(self, amount):
        """Generate random spectra
        
        Args:
            amount (int): Number of spectra
            wavenumbers (np.array): Wavenumbers
            peaks (np.array): Peak centers
            gamma (np.array): Peak widths
            eta (np.array): Mixing parameters
            alpha (np.array): Peak heights
        """

        # peaks = np.array([250])
        gamma = np.array([20])
        eta = np.array([0.5])
        alpha = np.array([1])

        #  Randomize peaks with shape (amount, len(peaks))
        peaks = np.random.uniform(100, 400, (amount, len(alpha)))

        # print(alpha.shape)

        random_spectra = np.zeros((amount, self.wavenumbers))
        for i in range(amount):
            ps = self.generate_full_spectrum(peaks[i], gamma, eta, alpha)
            random_spectra[i] = ps

        return random_spectra


if __name__ == "__main__":



    wavenumbers = 500
    ps = pseudoVoigtSimulatorTorch(wavenumbers)
    # s = ps.generate_full_spectrum(np.array([250]), np.array([20]), np.array([0.5]), np.array([1]), noise_to_signal_ratio=0)
    # print(s.shape)


    # ps_torch = pseudoVoigtSimulatorTorch(wavenumbers)
    # s_torch = ps_torch.generate_full_spectrum(torch.tensor([250]), torch.tensor([20]), torch.tensor([0.5]), torch.tensor([1]), noise_to_signal_ratio=0)
    # print(s_torch.shape)


    s= ps.generate_full_spectrum(torch.tensor([-1000]), torch.tensor([20]), torch.tensor([0.5]), torch.tensor([1]), sigma=0.5)
    print(s)


    # W = 500
    # ws = torch.arange(W)
    # c, gamma, eta = torch.tensor([-1000]), torch.tensor([20]), torch.tensor([0.5])

    # K = 1 if not c.shape else c.shape[0] 


    # wavenumbers = torch.tile(ws, (K, 1))
    # cs = torch.tile(c, (W, 1)).T
    # gammas = torch.tile(gamma, (W, 1)).T
    # etas = torch.tile(eta, (W, 1)).T


    # l = ps.lorentzian(ws, cs, gammas)
    # print(l)
    # g = ps.gaussian(ws, cs, gammas)
    # print(g)
    # # w, c, gamma, height_nomralize=True


    # Gs = 1/(np.sqrt(2*torch.pi)*gamma) * torch.exp((-(ws-cs)**2)/(2*gammas**2))
    # # divide by the maximum value
    # print(Gs)
    # Gs = Gs / torch.max(Gs)

    # print(Gs)
    # peaks = torch.tensor([250])
    # gamma = torch.tensor([20])
    # eta = torch.tensor([0.5])
    # alpha = torch.tensor([0.5])

    # for i, (s, y) in enumerate(ps.generator(1, peaks = (50,450),
    #                 gamma = torch.tensor([20]), eta = torch.tensor([0.5]), alpha = (0.5,10), sigma = 0.5)):
    #     print(y)

    #     break
    #     plt.plot(s)
    #     if i == 10:
    #         break

    # plt.show()
    





    # # peaks = np.array([250,350])
    # # gamma = np.array([20,20])
    # # eta = np.array([0.5,0.5])
    # # alpha = np.array([1,1])

    # s = ps.generate_full_spectrum(peaks, gamma, eta, alpha, sigma = 0.5)

    # plt.plot(s)
    # plt.show()

    # n_train = 100
    # train_test_split = 0.8
    # rs = ps.generate_random_spectra(1000)
    # rs_test = ps.generate_random_spectra(int(n_train/train_test_split * (1 - train_test_split)))

    # for i in range(10):
            
    #     plt.plot(rs[i])

    # plt.show()
    # # save rs to file 
    # np.save("random_spectra_peakr.npy", rs)
    # np.save("random_spectra_peakr_test.npy", rs_test)

