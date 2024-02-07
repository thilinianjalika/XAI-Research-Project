import commonStyles from "@/styles/commonStyles";
import { Box, IconButton, Paper } from "@mui/material";
import styles from "./styles";
import { Close } from "@mui/icons-material";

const ModalContainer = ({ children, show, close }) => {
    return (
        <Box sx={commonStyles.backdrop(show)} onClick={close}>
            <Paper sx={styles.modal} onClick={(e) => e.stopPropagation()}>
                <Box sx={styles.headRibbon}>
                    <IconButton sx={styles.cross} onClick={close}>
                        <Close />
                    </IconButton>
                </Box>
                <Box sx={styles.body}>{children}</Box>
            </Paper>
        </Box>
    );
};

export default ModalContainer;
