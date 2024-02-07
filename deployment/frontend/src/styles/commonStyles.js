import { alpha } from "@mui/material";

export default {
    btn: {
        margin: (theme) => theme.spacing(2),
    },
    btnContainer: {
        textAlign: "center",
        ".MuiButton-root": {
            margin: (theme) => theme.spacing(1),
        },
    },
    autocomplete: {
        margin: (theme) => `${theme.spacing(2)} 0`,
        width: "100%",
    },
    text: {
        margin: (theme) => `${theme.spacing(2)} 0`,
    },
    codeBlock: {
        fontFamily: "monospace",
        whiteSpace: "pre",
        overflowX: "scroll",
    },
    sectionContainer: {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        border: (theme) => `1px solid ${theme.palette.text.primary}`,
        padding: "30px",
        margin: "10px",
        width: "100%",
        borderRadius: "10px",
        ".MuiTypography-h3": {
            margin: "15px 0px",
            width: "100%",
        },
        ".MuiTypography-h4": {
            margin: "10px 0px",
            width: "100%",
        },
    },
    outputContainer: (ok) => ({
        position: "relative",
        border: (theme) =>
            `1px solid ${
                ok === undefined
                    ? theme.palette.text.icon
                    : ok
                    ? theme.palette.success.main
                    : theme.palette.error.main
            }`,
        width: "100%",
        padding: (theme) => theme.spacing(2),
        borderRadius: (theme) => theme.spacing(1),
        color: (theme) =>
            ok === undefined
                ? theme.palette.text.icon
                : ok
                ? theme.palette.success.main
                : theme.palette.error.main,
    }),
    backdrop: (active) => ({
        display: active ? "flex" : "none",
        justifyContent: "center",
        alignItems: "center",
        position: "fixed",
        width: "100vw",
        height: "100vh",
        left: 0,
        top: 0,
        backgroundColor: (theme) =>
            alpha(theme.palette.background.default, 0.5),
        zIndex: 1,
    }),
};
