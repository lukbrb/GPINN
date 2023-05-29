# Calculs potentiel Dehnen

Pour trouver une primitive de cette fonction, nous pouvons utiliser une méthode appelée substitution. Nous allons poser u = 1+r, ce qui nous donne du = dr.

En remplaçant dans notre expression, nous avons :

\begin{align*}
\int \dfrac{r^{2-a}}{(1+r)^{4-a}} dr &= \int \dfrac{(u-1)^{2-a}}{u^{4-a}} du \
&= \int \dfrac{u^{a-3}(u-1)^{2-a}}{u^4} du \
&= \int \dfrac{u^{-2}}{(1-\frac{1}{u})^{a-2}} du.
\end{align*}

Nous pouvons ensuite faire une autre substitution, en posant v = 1 - 1/u, ce qui nous donne dv = 1/u^2 du. En remplaçant dans notre expression, nous avons :

\begin{align*}
\int \dfrac{u^{-2}}{(1-\frac{1}{u})^{a-2}} du &= \int \dfrac{v^{a-4}}{(1-v)^{a-2}} dv \
&= \int \dfrac{v^{a-4}}{(1-v)^{a-2}} \dfrac{(1-v)^{3-a}}{(3-a)(-1)} dv \
&= -\dfrac{1}{(a-3)} \int v^{-2} (1-v)^{3-a} dv.
\end{align*}

Nous pouvons maintenant intégrer cette dernière expression en utilisant une intégration par parties, en posant u = (1-v)^{4-a} et dv = v^{-2} dv. Nous avons alors du = -(4-a)(1-v)^{3-a}(-1) dv.

En remplaçant dans notre expression, nous avons :

\begin{align*}
-\dfrac{1}{(a-3)} \int v^{-2} (1-v)^{3-a} dv &= -\dfrac{1}{(a-3)} \int u \dfrac{du}{-(4-a)(1-v)^{3-a}} \
&= \dfrac{1}{(a-3)(a-4)} u + C \
&= \dfrac{(1-v)^{4-a}}{(a-3)(a-4)} + C,
\end{align*}

où C est une constante d'intégration. En remplaçant u et v par leurs expressions d'origine, nous avons finalement :

\begin{align*}
\int \dfrac{r^{2-a}}{(1+r)^{4-a}} dr &= \dfrac{(1+r)^{3-a}}{(a-3)(a-4)} + C.
\end{align*}


\begin{align*}
\int \dfrac{(u-1)^{2-a}}{u^{4-a}} du &= \int \dfrac{u^{2-a}-2u^{1-a}+1}{u^{4-a}} du \
&= \int u^{-2} du - 2\int u^{-3+a} du + \int u^{-4+a} du \
&= -\dfrac{u^{-1}}{a-3} + \dfrac{u^{a-3}}{(a-3)(2-a)} - \dfrac{u^{a-4}}{(a-3)(3-a)} + C,
\end{align*}

Pour résoudre cette équation en utilisant une fonction de Green, nous pouvons suivre les étapes suivantes :

- Définir les conditions aux limites appropriées pour l'équation. Dans ce cas, nous avons besoin de deux conditions aux limites : une condition à la limite intérieure, par exemple $\phi(0) = 0$, et une condition à la limite extérieure, par exemple $\phi(R) = 1$, où $R$ est le rayon de la région à l'intérieur de laquelle nous cherchons la solution.

- Trouver la fonction de Green pour cette équation. La fonction de Green est une fonction qui satisfait l'équation homogène associée à l'équation originale, avec les mêmes conditions aux limites, mais avec une source ponctuelle à une position arbitraire. Dans ce cas, l'équation homogène associée est $\dfrac{d}{dr}\left(r^2\dfrac{d G}{dr}\right) = 0$, qui a pour solution générale $G(r,r') = c_1\ln(r/r') + c_2$, où $c_1$ et $c_2$ sont des constantes déterminées par les conditions aux limites. En utilisant les conditions aux limites, on peut trouver que $c_1 = -\ln(R)$ et $c_2 = 1$. Ainsi, la fonction de Green est donnée par $G(r,r') = \ln(r'/r) - \ln(R)$.

- Utiliser la fonction de Green pour trouver la solution de l'équation originale. La solution de l'équation originale est donnée par $\phi(r) = \int_0^R G(r,r') f(r')dr'$, où $f(r') = \dfrac{2r'^{2-\gamma}}{(1+r')^{4-\gamma}}$ est la source ponctuelle. En effectuant cette intégration, on obtient finalement la solution suivante :

\begin{align*}
\phi(r) &= \frac{1}{\ln(R)}\left(\ln\left(\frac{r}{R}\right)\int_0^r \frac{2r'^{2-\gamma}}{(1+r')^{4-\gamma}}\ln\left(\frac{r'}{R}\right)dr' \right.\
&\quad\left.-\int_r^R \frac{2r'^{2-\gamma}}{(1+r')^{4-\gamma}}\ln\left(\frac{r}{r'}\right)dr'\right) + 1
\end{align*}
