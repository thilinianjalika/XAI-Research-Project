import ModalProvider from "@/providers/modalProvider/ModalProvider";
import "@/styles/globals.css";
import { darkTheme, lightTheme } from "@/styles/themes";
import { ThemeProvider } from "@mui/material";
import { useEffect, useState } from "react";

export default function App({ Component, pageProps }) {
    const [theme, setTheme] = useState(lightTheme);
    useEffect(() => {
        if (
            window.matchMedia &&
            window.matchMedia("(prefers-color-scheme: dark)").matches
        ) {
            setTheme(darkTheme);
        }
    }, []);

    return (
        <ThemeProvider theme={theme}>
            <ModalProvider>
                <Component {...pageProps} />
            </ModalProvider>
        </ThemeProvider>
    );
}
