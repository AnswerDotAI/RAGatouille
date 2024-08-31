def delete_from_index(
    self,
    document_ids: Union[TypeVar("T"), List[TypeVar("T")]],
    index_name: Optional[str] = None,
):
    self.index_name = index_name if index_name is not None else self.index_name
    if self.index_name is None:
        print(
            "Cannot delete from index without an index_name! Please provide one.",
            "Returning empty results.",
        )
        return None

    print(
        "WARNING: delete_from_index support is currently experimental!",
        "delete_from_index support will be more thorough in future versions",
    )

    pids_to_remove = []
    for pid, docid in self.pid_docid_map.items():
        if docid in document_ids:
            pids_to_remove.append(pid)

    # TODO We may want to load an existing index here instead;
    #      For now require that either index() was called, or an existing one was loaded.
    assert self.model_index is not None

    # TODO We probably want to store some of this in the model_index directly.
    self.model_index.delete(
        self.config,
        self.checkpoint,
        self.collection,
        self.index_name,
        pids_to_remove,
        verbose=self.verbose != 0,
    )

    # Update and serialize the index metadata + collection.
    self.collection = [
        doc for pid, doc in enumerate(self.collection) if pid not in pids_to_remove
    ]
    self.pid_docid_map = {
        pid: docid
        for pid, docid in self.pid_docid_map.items()
        if pid not in pids_to_remove
    }

    if self.docid_metadata_map is not None:
        self.docid_metadata_map = {
            docid: metadata
            for docid, metadata in self.docid_metadata_map.items()
            if docid not in document_ids
        }

    self._save_index_metadata()

    print(f"Successfully deleted documents with these IDs: {document_ids}")

    def _write_collection_files_to_disk(
        self,
        overwrite_collection: bool = True,
        overwrite_pid_docid_map: bool = True,
        overwrite_docid_metadata_map: bool = True,
    ):
        if overwrite_collection:
            srsly.write_json(self.index_path + "/collection.json", self.collection)
        if overwrite_pid_docid_map:
            srsly.write_json(
                self.index_path + "/pid_docid_map.json", self.pid_docid_map
            )
        if self.docid_metadata_map is not None and overwrite_docid_metadata_map:
            srsly.write_json(
                self.index_path + "/docid_metadata_map.json", self.docid_metadata_map
            )

        # update the in-memory inverted map every time the files are saved to disk
        self.docid_pid_map = self._invert_pid_docid_map()

    def _save_index_metadata(self):
        assert self.model_index is not None

        model_metadata = srsly.read_json(self.index_path + "/metadata.json")
        index_config = self.model_index.export_metadata()
        index_config["index_name"] = self.index_name
        # Ensure that the additional metadata we store does not collide with anything else.
        model_metadata["RAGatouille"] = {"index_config": index_config}  # type: ignore
        srsly.write_json(self.index_path + "/metadata.json", model_metadata)
        self._write_collection_files_to_disk()
