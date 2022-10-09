# Here's my sample title

This is some sample text.

(section-label)=
## Here's my first section

Here is a [reference to the intro](intro.md). Here is a reference to [](section-label).

I am going to test out writing a note inside an equation here:
```{note}
This is a note!
```

Here's one equation:

\begin{gather*}
a_1=b_1+c_1\\
a_2=b_2+c_2-d_2+e_2
\end{gather*}


And then an aligned envrionment:

\begin{align}
a_{11}& =b_{11}&
  a_{12}& =b_{12}\\
a_{21}& =b_{21}&
  a_{22}& =b_{22}+c_{22}
\end{align}


```{math}
:label: my_other_label
w_{t+1} = (1 + r_{t+1}) s(w_t) + y_{t+1}
```

See Eq. {eq}`my_other_label` for some details.