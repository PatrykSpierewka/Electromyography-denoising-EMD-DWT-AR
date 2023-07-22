# EMG-denoising-EMD-DWT-AR
In order to obtain the noisy EMG signal needed for denoising, a set of signals from the PhysioNet website was selected:
- https://physionet.org/content/emgdb/1.0.0/
- https://physionet.org/content/hd-semg/1.0.0/

<p align="center">
    <img width="647" alt="Zrzut ekranu 2023-07-22 o 18 09 41" src="https://github.com/PatrykSpierewka/Electromyography-denoising-EMD-DWT-AR/assets/101202344/8c30211a-4670-4e41-9909-6a66efa29b66">
</p>

Three algorithms were used for denoising:
- Empirical Mode Decomposition,
- Discrete Wavelet Transform,
- Autoregressive Model.


<p align="center">
    <img width="647" alt="Zrzut ekranu 2023-07-22 o 18 19 17" src="https://github.com/PatrykSpierewka/Electromyography-denoising-EMD-DWT-AR/assets/101202344/a28b8df9-26fb-4723-a5a8-93ca7dbd9598">
</p>

The quality of the denoise was determined by a reference signal that had two versions: denoised and noised. The original denoised signal was compared to the denoised signal using the following algorithms: EMD, DWT, AR. Qualitative measures such as Mean Squared Error, Mean Absolute Error and Signal to Noise Ratio were calculated.

<p align="center">
    <img width="385" alt="Zrzut ekranu 2023-07-22 o 18 27 26" src="https://github.com/PatrykSpierewka/Electromyography-denoising-EMD-DWT-AR/assets/101202344/3e82fda5-de85-4706-a9a3-78967cc78787">
</p>

<p align="center">
    <img width="736" alt="Zrzut ekranu 2023-07-22 o 18 28 13" src="https://github.com/PatrykSpierewka/Electromyography-denoising-EMD-DWT-AR/assets/101202344/38a9c542-402f-4bf6-a38b-4cb9c79903b9">
</p>
