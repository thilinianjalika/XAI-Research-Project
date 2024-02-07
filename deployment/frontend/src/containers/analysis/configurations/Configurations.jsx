import { Box, IconButton, Typography } from "@mui/material";
import styles from "./configForm/styles";
import { Add } from "@mui/icons-material";
import { useState } from "react";
import ConfigForm from "./configForm/ConfigForm";
import ConfigTable from "./configTable/ConfigTable";
import ModalContainer from "@/components/modalContainer/ModalContainer";

const Configurations = ({ model, configurations, setConfigurations }) => {
    const [displayModal, setDisplayModal] = useState(false);
    const addConfig = (newConfig) => {
        setConfigurations([...configurations, newConfig]);
        setDisplayModal(false);
    };
    return (
        <Box sx={styles.root}>
            {configurations.length !== 0 && (
                <ConfigTable
                    configurations={configurations}
                    setConfigurations={setConfigurations}
                    model={model}
                />
            )}
            <IconButton
                onClick={() => setDisplayModal(true)}
                sx={styles.addRow}
            >
                <Add />
            </IconButton>
            <ModalContainer
                show={displayModal}
                close={() => setDisplayModal(false)}
            >
                <Typography variant="h4">Add Test Case</Typography>
                <ConfigForm
                    model={model}
                    addConfig={addConfig}
                    close={() => setDisplayModal(false)}
                    displayModal={displayModal}
                />
            </ModalContainer>
        </Box>
    );
};

export default Configurations;
