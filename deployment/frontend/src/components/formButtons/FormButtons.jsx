import commonStyles from "@/styles/commonStyles";
import { Box, Button } from "@mui/material";

const FormButtons = ({ reset, close }) => {
    return (
        <Box sx={commonStyles.btnContainer}>
            <Button variant="outlined" type="submit">
                Add
            </Button>
            <Button variant="outlined" onClick={reset}>
                Reset
            </Button>
            <Button variant="outlined" onClick={close}>
                Close
            </Button>
        </Box>
    );
};

export default FormButtons;
