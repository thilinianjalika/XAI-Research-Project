import { alpha } from "@mui/material";

export default {
    root: {
        width: "100%",
    },
    addRow: {
        border: (theme) =>
            `1px dashed ${alpha(theme.palette.text.primary, 0.5)}`,
        textAlign: "center",
        width: "100%",
        borderRadius: "10px",
    },
    modalRoot: {
        display: "flex",
        flexDirection: "column",
    },
};
